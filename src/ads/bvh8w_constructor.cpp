/*
 *
 * wave tracer
 * Copyright  Shlomi Steinberg
 *
 * LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
 *
 */

#include <array>
#include <future>
#include <vector>
#include <memory>

#include <wt/ads/bvh8w/bvh8w_constructor.hpp>
#include <wt/ads/bvh8w/common.hpp>
#include <wt/ads/edge_classification.hpp>
#include <wt/ads/bvh_constructor.hpp>
#include <wt/ads/traversal_common.hpp>

#include <wt/util/thread_pool/tpool.hpp>

using namespace wt;
using namespace wt::ads;
using namespace wt::ads::construction;


template <int depth>
inline void extract_bvh_nodes(const bvh_t::node_t& bvn,
                              const std::vector<bvh_t::node_cluster_t>& bvnc,
                              std::vector<const bvh_t::node_t*>& nodes,
                              int& nodes_consumed) noexcept {
    if (bvn.is_leaf || depth>0)
        ++nodes_consumed;
    if (bvn.is_leaf || depth==0) {
        nodes.emplace_back(&bvn);
        return;
    }

    if constexpr (depth>0) {
        extract_bvh_nodes<depth-1>(bvnc[bvn.children_cluster_offset].nodes[0], bvnc, nodes, nodes_consumed);
        extract_bvh_nodes<depth-1>(bvnc[bvn.children_cluster_offset].nodes[1], bvnc, nodes, nodes_consumed);
    }
}


// One 8-wide node: up to 8 nodes from the 3 binary levels below a binary node.
struct w8_build_t {
    std::array<const bvh_t::node_t*, 8> children;
    std::uint8_t count = 0;
    int bbvh_nodes_consumed = 0;
};

inline void build_node8w(bvh8w::node_t& w8node,
                         w8_build_t& out,
                         const bvh::node_t& bvn,
                         const std::vector<bvh_t::node_cluster_t>& bvnc,
                         const std::vector<tri_t>& tris) noexcept {
    // extract 3 levels of nodes from the BVH
    std::vector<const bvh_t::node_t*> bvh_child_nodes;
    bvh_child_nodes.reserve(8);
    extract_bvh_nodes<3>(bvn, bvnc, bvh_child_nodes, out.bbvh_nodes_consumed);

    std::array<length_t,8> node_min_x, node_max_x, node_min_y, node_max_y, node_min_z, node_max_z;

    // encode
    for (auto c=0ul; c<bvh_child_nodes.size(); ++c) {
        const auto& n = bvh_child_nodes[c];
        node_min_x[c] = n->aabb.min.x;
        node_min_y[c] = n->aabb.min.y;
        node_min_z[c] = n->aabb.min.z;
        node_max_x[c] = n->aabb.max.x;
        node_max_y[c] = n->aabb.max.y;
        node_max_z[c] = n->aabb.max.z;

        out.children[c] = n;
    }
    out.count = (std::uint8_t)bvh_child_nodes.size();

    w8node.min = pqvec3_w8_t{ node_min_x.data(), node_min_y.data(), node_min_z.data(), simd::unaligned_data };
    w8node.max = pqvec3_w8_t{ node_max_x.data(), node_max_y.data(), node_max_z.data(), simd::unaligned_data };
    w8node.tris_start = bvn.tris_offset;
    w8node.tris_count = bvn.tri_count;

#ifdef DEBUG
    // sanity
    for (auto t = w8node.tris_start; t<w8node.tris_start+w8node.tris_count; ++t) {
        const auto& tri = tris[t];
        assert(bvn.aabb.contains<range_inclusiveness_e::inclusive>(tri.a));
        assert(bvn.aabb.contains<range_inclusiveness_e::inclusive>(tri.b));
        assert(bvn.aabb.contains<range_inclusiveness_e::inclusive>(tri.c));
    }
#endif
}

void calculate_occupancy(const bvh8w::node_t& n, const std::vector<bvh8w::node_t>& nodes,
                         std::size_t& filled_chld, std::size_t& potential_chld, std::size_t& max_depth, std::size_t depth=0) {
    potential_chld += 8;
    max_depth = std::max(max_depth, depth);
    for (const auto& ptr : n.child_ptrs) {
         if (!bvh8w::is_ptr_empty(ptr)) {
            ++filled_chld;
            if (bvh8w::is_ptr_child(ptr))
                calculate_occupancy(nodes[bvh8w::child_node_ptr(ptr)],
                                    nodes, filled_chld, potential_chld, max_depth, depth+1);
        }
    }
}

inline auto create_w8_tris(const std::vector<tri_t>& tris) {
    bvh8w_t::tris_vectorized_data_t d;
    const auto ds = tris.size()+7;
    d.ax.resize(ds);
    d.ay.resize(ds);
    d.az.resize(ds);
    d.bx.resize(ds);
    d.by.resize(ds);
    d.bz.resize(ds);
    d.cx.resize(ds);
    d.cy.resize(ds);
    d.cz.resize(ds);
    d.nx.resize(ds);
    d.ny.resize(ds);
    d.nz.resize(ds);

    for (auto t=0ul;t<tris.size();++t) {
        d.ax[t] = tris[t].a.x;
        d.ay[t] = tris[t].a.y;
        d.az[t] = tris[t].a.z;
        d.bx[t] = tris[t].b.x;
        d.by[t] = tris[t].b.y;
        d.bz[t] = tris[t].b.z;
        d.cx[t] = tris[t].c.x;
        d.cy[t] = tris[t].c.y;
        d.cz[t] = tris[t].c.z;
        d.nx[t] = tris[t].n.x;
        d.ny[t] = tris[t].n.y;
        d.nz[t] = tris[t].n.z;
    }

    return d;
}

bvh8w_constructor_t::bvh8w_constructor_t(
        std::vector<std::shared_ptr<shape_t>> objs,
        const wt::wt_context_t& ctx,
        std::optional<progress_callback_t> progress_callbacks)
{
    constexpr f_t pt_bvh_progress_portion = .7;
    constexpr f_t pt_8w_progress_portion  = .2;
    constexpr f_t pt_edge_finding_portion = 1 - pt_8w_progress_portion - pt_bvh_progress_portion;


    const auto start_timepoint = std::chrono::high_resolution_clock::now();
    pt.callbacks = std::move(progress_callbacks);


    // build plain binary BVH
    pt.proportion = pt_bvh_progress_portion;

    auto bvh_ctor = bvh_constructor_t{ std::move(objs), ctx, pt };
    auto bvh = std::move(bvh_ctor.bvh);

    if (!bvh)
        throw std::runtime_error("(BVH8w) BVH construction failed");

    const auto& bvhnc = bvh->node_clusters();


    // ... and encode an 8-wide BVH from it

    pt.start = pt_bvh_progress_portion;
    pt.proportion = pt_8w_progress_portion;
    pt.set_status("encoding 8-wide BVH");

    const auto total = bvhnc.size()*2;
    std::size_t completed = 0;

    std::vector<bvh8w::node_t> w8nodes;
    std::vector<bvh8w::leaf_node_t> w8_leaf_nodes;
    w8nodes.reserve(total/8);

    // create 8-wide clusters in parallel
    auto w8tris_future = std::async(std::launch::async, [&](){ return create_w8_tris(bvh->tree_tris); });

    // Encode the 8-wide tree breadth first, a level at a time. A level's nodes are built
    // in parallel; then, in order, their inner children get the next level's indices and
    // their leaves the next leaf indices.
    const auto world = bvh->V();
    std::vector<const bvh_t::node_t*> level = { &bvh->root() };
    std::vector<w8_build_t> built;
    constexpr std::size_t chunk = 4096;

    while (!level.empty()) {
        const auto first = w8nodes.size();
        w8nodes.resize(first + level.size());
        built.assign(level.size(), w8_build_t{});

        const auto build_range = [&](std::size_t begin, std::size_t end) {
            for (auto i=begin; i<end; ++i)
                build_node8w(w8nodes[first+i], built[i], *level[i], bvhnc, bvh->tree_tris);
        };
        if (level.size()<=chunk) {
            build_range(0, level.size());
        } else {
            std::vector<std::future<void>> chunks;
            for (auto begin=0ul; begin<level.size(); begin+=chunk)
                chunks.emplace_back(ctx.threadpool->enqueue([&, begin]() {
                    build_range(begin, std::min(begin+chunk, level.size()));
                }));
            for (auto& f : chunks)
                f.get();
        }

        std::vector<const bvh_t::node_t*> next;
        for (auto i=0ul; i<level.size(); ++i) {
            auto& w8node = w8nodes[first+i];
            const auto& b = built[i];
            for (auto c=0; c<b.count; ++c) {
                const auto& n = *b.children[c];
                if (n.is_leaf) {
                    w8_leaf_nodes.emplace_back(bvh8w::leaf_node_t{ .tris_ptr = n.tris_offset, .count = n.tri_count });
                    if (w8_leaf_nodes.size()>limits<std::int32_t>::max())       // panic
                        throw std::runtime_error("(BVH8w) Too many nodes!");

                    w8node.child_ptrs[c] = -(std::int32_t)(w8_leaf_nodes.size());  // leaf ptrs have set signs
                } else {
                    const auto cidx = first + level.size() + next.size();
                    if (cidx>limits<std::int32_t>::max())       // panic
                        throw std::runtime_error("(BVH8w) Too many nodes!");

                    w8node.child_ptrs[c] = (std::int32_t)cidx+1;   // child ptrs have unset signs
                    next.push_back(&n);
                }
            }
            completed += b.bbvh_nodes_consumed;
        }
        pt.set_progress(completed / f_t(total));
        level = std::move(next);
    }
    assert(completed==total-1);

    // calculate occupancy
    std::size_t filled_chld = 0, potential_chld = 0, max_depth = 0;
    calculate_occupancy(w8nodes[0], w8nodes, filled_chld, potential_chld, max_depth);
    f_t occupancy = filled_chld / f_t(potential_chld);

    // finish tri clusters
    auto w8tris = std::move(w8tris_future).get();


    // create bvh8w
    w8nodes.shrink_to_fit();
    w8_leaf_nodes.shrink_to_fit();
    bvh8w = std::make_unique<bvh8w_t>(std::move(w8nodes), std::move(w8_leaf_nodes), 
                                      std::move(w8tris), std::move(bvh->tree_tris), 
                                      world, bvh->get_sah_cost(), occupancy, max_depth);


    // find edges
    pt.start = pt_bvh_progress_portion + pt_8w_progress_portion;
    pt.proportion = pt_edge_finding_portion;
    pt.set_status("finding edges");

    bvh8w->edges = find_edges(bvh8w.get(), bvh8w->tris, ctx, pt);
    

    // done building
    this->build_time = std::chrono::high_resolution_clock::now() - start_timepoint;
    pt.set_status("");
    pt.complete();
}
