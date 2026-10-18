/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#include <wt/gpu/gpu.hpp>
#include <wt/gpu/backend.hpp>

#include <algorithm>

#include <wt/ads/bvh8w/bvh8w.hpp>

#include "to_kernel.hpp"

using namespace wt;
using namespace wt::gpu;

namespace {

using kernel::real;

bvh_arrays_t flatten(const ads::bvh8w_t& bvh) {
    bvh_arrays_t arrays;

    arrays.nodes.resize(bvh.nodes_count());
    for (std::size_t i=0; i<arrays.nodes.size(); ++i) {
        const auto& n = bvh.node(ads::idx_t(i));
        auto& k = arrays.nodes[i];
        for (std::size_t c=0; c<kernel::bvh_width; ++c) {
            const auto mn = vec3_t{ u::to_m(n.min.read(c)) };
            const auto mx = vec3_t{ u::to_m(n.max.read(c)) };
            k.minx[c] = real(mn.x); k.miny[c] = real(mn.y); k.minz[c] = real(mn.z);
            k.maxx[c] = real(mx.x); k.maxy[c] = real(mx.y); k.maxz[c] = real(mx.z);
            k.child[c] = n.child_ptrs[c];
        }
        k.tris_start = n.tris_start;
        k.tris_count = n.tris_count;
    }

    arrays.leaves.resize(bvh.leaf_nodes_count());
    for (std::size_t i=0; i<arrays.leaves.size(); ++i) {
        const auto& l = bvh.leaf_node(ads::idx_t(i));
        arrays.leaves[i] = { l.tris_ptr, l.count };
    }

    arrays.tris.resize(bvh.triangles_count());
    for (std::size_t i=0; i<arrays.tris.size(); ++i) {
        const auto& t = bvh.tri(ads::tuid_t{ ads::idx_t(i) });
        arrays.tris[i] = { to_kernel(t.a), to_kernel(t.b), to_kernel(t.c), to_kernel(t.n) };
    }

    return arrays;
}

void check_status(std::uint32_t status) {
    if (status & kernel::status_stack_overflow)
        throw gpu_error_t("GPU BVH traversal overflowed its stack");
}

/** @brief Queries per backend call, bounding device memory for results. */
constexpr std::size_t batch_size = std::size_t(1)<<16;
/** @brief Triangles per cone or ball query on the first attempt; queries finding more are re-run. */
constexpr std::uint32_t initial_triangle_capacity = 32;

std::uint32_t tri_count(const kernel::cone_hit& h) noexcept { return h.tri_count; }
std::uint32_t tri_count(const ball_hit_t& h) noexcept { return h.tri_count; }

/**
 * @brief Runs queries returning triangle lists, `run(queries, count, capacity, hits, tris)`,
 *        re-running those that found more triangles than the capacity with enough room.
 */
template <typename Query, typename Hit, typename Run>
void collect_triangle_lists(const std::vector<Query>& queries, const Run& run,
                            std::vector<Hit>& hits,
                            geometry_t::triangle_lists_t& lists) {
    hits.resize(queries.size());
    lists.offsets.assign(1, 0);
    lists.offsets.reserve(queries.size()+1);

    std::vector<std::uint32_t> tris, retry_tris;
    std::vector<Query> retry_queries;
    std::vector<Hit> retry_hits;
    std::vector<std::size_t> retry_slot;

    for (std::size_t start=0; start<queries.size(); start+=batch_size) {
        const auto count = std::min(batch_size, queries.size()-start);
        const auto capacity = initial_triangle_capacity;
        tris.resize(count*capacity);
        run(queries.data()+start, count, capacity, hits.data()+start, tris.data());

        // re-run the queries that did not fit
        retry_queries.clear();
        retry_slot.assign(count, std::size_t(-1));
        std::uint32_t retry_capacity = 0;
        for (std::size_t i=0; i<count; ++i) {
            if (tri_count(hits[start+i])>capacity) {
                retry_slot[i] = retry_queries.size();
                retry_queries.push_back(queries[start+i]);
                retry_capacity = std::max(retry_capacity, tri_count(hits[start+i]));
            }
        }
        if (!retry_queries.empty()) {
            retry_hits.resize(retry_queries.size());
            retry_tris.resize(retry_queries.size()*retry_capacity);
            run(retry_queries.data(), retry_queries.size(), retry_capacity,
                retry_hits.data(), retry_tris.data());
        }

        for (std::size_t i=0; i<count; ++i) {
            const std::uint32_t* row;
            if (retry_slot[i]==std::size_t(-1)) {
                row = &tris[i*capacity];
            } else {
                const auto& retry = retry_hits[retry_slot[i]];
                if (tri_count(retry)!=tri_count(hits[start+i]))
                    throw gpu_error_t("GPU query returned different results when re-run");
                row = &retry_tris[retry_slot[i]*retry_capacity];
            }
            check_status(hits[start+i].status);
            for (std::uint32_t t=0; t<tri_count(hits[start+i]); ++t)
                lists.triangles.push_back(ads::tuid_t{ row[t] });
            lists.offsets.push_back(lists.triangles.size());
        }
    }
}

}


std::string wt::gpu::compiled_backends() {
#if defined(WT_GPU_CUDA) && defined(WT_GPU_OPTIX)
    return "CUDA, OptiX";
#elif defined(WT_GPU_CUDA)
    return "CUDA";
#else
    return "";
#endif
}


/*
 * device_t
 */

device_t::device_t(std::unique_ptr<backend_t> backend) noexcept : backend(std::move(backend)) {}
device_t::~device_t() noexcept = default;

std::unique_ptr<device_t> device_t::create(const options_t& options) {
#if defined(WT_GPU_CUDA)
    try {
        return std::unique_ptr<device_t>(new device_t(create_cuda_backend(options.ordinal, options.use_optix)));
    } catch (const std::exception& e) {
        throw gpu_error_t(std::string("cannot use the GPU: ") + e.what());
    }
#else
    (void)options;
    throw gpu_error_t("this build of wave_tracer has no GPU backend (configure it with -DGPU_CUDA=ON)");
#endif
}

std::string device_t::description() const {
    return backend->description();
}

std::unique_ptr<geometry_t> device_t::upload(const ads::ads_t& ads) const {
    const auto* bvh = dynamic_cast<const ads::bvh8w_t*>(&ads);
    if (!bvh)
        throw gpu_error_t("the GPU supports only the bvh8w acceleration structure");
    try {
        return std::unique_ptr<geometry_t>(new geometry_t(backend->upload(flatten(*bvh))));
    } catch (const gpu_error_t&) {
        throw;
    } catch (const std::exception& e) {
        throw gpu_error_t(std::string("cannot upload the scene to the GPU: ") + e.what());
    }
}


/*
 * geometry_t
 */

geometry_t::geometry_t(std::unique_ptr<backend_geometry_t> backend) noexcept : backend(std::move(backend)) {}
geometry_t::~geometry_t() noexcept = default;

std::vector<std::optional<geometry_t::ray_hit_t>> geometry_t::intersect(
        std::span<const ray_t> rays, const pqrange_t<>& range) const {
    std::vector<ray_query_t> queries(rays.size());
    const auto krange = to_kernel(range);
    for (std::size_t i=0; i<rays.size(); ++i)
        queries[i] = { to_kernel(rays[i]), krange };

    std::vector<kernel::ray_hit> hits(rays.size());
    {
        const std::scoped_lock lock(mutex);
        for (std::size_t start=0; start<rays.size(); start+=batch_size)
            backend->intersect_rays(queries.data()+start, std::min(batch_size, rays.size()-start),
                                    hits.data()+start);
    }

    std::vector<std::optional<ray_hit_t>> ret(rays.size());
    for (std::size_t i=0; i<rays.size(); ++i) {
        const auto& h = hits[i];
        check_status(h.status);
        if (!h.hit) continue;
        ret[i] = ray_hit_t{
            .intersection = {
                .dist = f_t(h.dist) * u::m,
                .bary = barycentric_t(vec2_t{ f_t(h.baryx), f_t(h.baryy) }),
            },
            .tuid = ads::tuid_t{ h.tri },
            .front_face = h.front_face!=0,
        };
    }
    return ret;
}

std::vector<bool> geometry_t::shadow(std::span<const ray_t> rays, const pqrange_t<>& range) const {
    std::vector<ray_query_t> queries(rays.size());
    const auto krange = to_kernel(range);
    for (std::size_t i=0; i<rays.size(); ++i)
        queries[i] = { to_kernel(rays[i]), krange };

    std::vector<std::uint8_t> occluded(rays.size());
    std::uint32_t status = kernel::status_ok;
    {
        const std::scoped_lock lock(mutex);
        for (std::size_t start=0; start<rays.size(); start+=batch_size)
            backend->rays_occluded(queries.data()+start, std::min(batch_size, rays.size()-start),
                                   occluded.data()+start, &status);
    }
    check_status(status);

    return { occluded.begin(), occluded.end() };
}

geometry_t::cone_hits_t geometry_t::intersect(
        std::span<const elliptic_cone_t> cones, const pqrange_t<>& range,
        f_t z_search_range_scale) const {
    std::vector<cone_query_t> queries(cones.size());
    const auto krange = to_kernel(range);
    for (std::size_t i=0; i<cones.size(); ++i)
        queries[i] = { to_kernel(cones[i]), krange, real(z_search_range_scale) };

    std::vector<kernel::cone_hit> hits;
    cone_hits_t ret;
    {
        const std::scoped_lock lock(mutex);
        collect_triangle_lists(queries,
            [&](const cone_query_t* q, std::size_t n, std::uint32_t capacity,
                kernel::cone_hit* h, std::uint32_t* tris) {
                backend->intersect_cones(q, n, capacity, h, tris);
            },
            hits, ret.triangles);
    }

    ret.distances.resize(cones.size());
    ret.front_faces.resize(cones.size());
    for (std::size_t i=0; i<cones.size(); ++i) {
        // as intersection_record_t: a query without triangles hit nothing
        const bool hit = hits[i].tri_count>0;
        ret.distances[i] = hit ? length_t{ f_t(hits[i].dist) * u::m } : limits<length_t>::infinity();
        ret.front_faces[i] = hit && hits[i].front_face!=0;
    }
    return ret;
}

geometry_t::triangle_lists_t geometry_t::intersect(std::span<const ball_t> balls) const {
    std::vector<kernel::ball> queries(balls.size());
    for (std::size_t i=0; i<balls.size(); ++i)
        queries[i] = to_kernel(balls[i]);

    std::vector<ball_hit_t> hits;
    triangle_lists_t ret;
    const std::scoped_lock lock(mutex);
    collect_triangle_lists(queries,
        [&](const kernel::ball* q, std::size_t n, std::uint32_t capacity,
            ball_hit_t* h, std::uint32_t* tris) {
            backend->intersect_balls(q, n, capacity, h, tris);
        },
        hits, ret);
    return ret;
}
