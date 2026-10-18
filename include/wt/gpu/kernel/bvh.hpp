/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include "intersect.hpp"

/*
 * GPU copy of wave_tracer's 8-wide BVH (wt::ads::bvh8w_t) and its traversals,
 * ported from src/ads/bvh8w.cpp: same node layout and pointer encoding, same
 * visiting order (children sorted by entry distance), the same search-range
 * shrinking for cones and the same small-subtree shortcuts, so a query returns
 * what the CPU returns (up to floating-point rounding).
 *
 * Host code builds these arrays from a bvh8w_t (src/gpu/geometry_upload.cpp).
 */

namespace wt::gpu::kernel {

constexpr int bvh_width = 8;

/** @brief Internal node: 8 child boxes (SoA) and child pointers. */
struct bvh_node {
    real minx[bvh_width], miny[bvh_width], minz[bvh_width];
    real maxx[bvh_width], maxy[bvh_width], maxz[bvh_width];
    /** 0: empty; >0: internal node index+1; <0: -(leaf index+1). */
    i32 child[bvh_width];
    /** Triangles of the whole subtree (contiguous). */
    u32 tris_start, tris_count;
};

struct bvh_leaf {
    u32 tris_ptr, count;
};

struct bvh_tri {
    vec3 a, b, c;
    vec3 n;         // geometric normal, as stored by the CPU (not recomputed)
};

/** @brief The uploaded BVH. The root is node 0. */
struct bvh_view {
    const WT_GPU_GLOBAL bvh_node* nodes;
    const WT_GPU_GLOBAL bvh_leaf* leaves;
    const WT_GPU_GLOBAL bvh_tri*  tris;
};

constexpr i32 bvh_root_ptr = 1;

WT_GPU_FN bool bvh_is_leaf(i32 ptr)  { return ptr<0; }
WT_GPU_FN u32  bvh_leaf_index(i32 ptr) { return u32((-ptr)-1); }
WT_GPU_FN u32  bvh_node_index(i32 ptr) { return u32(ptr-1); }

WT_GPU_FN vec3 bvh_child_min(const WT_GPU_GLOBAL bvh_node& n, int i) { return { n.minx[i], n.miny[i], n.minz[i] }; }
WT_GPU_FN vec3 bvh_child_max(const WT_GPU_GLOBAL bvh_node& n, int i) { return { n.maxx[i], n.maxy[i], n.maxz[i] }; }

/** @brief Nodes with at most this many triangles are tested as one leaf by ray and ball queries. */
constexpr u32 bvh_treat_node_as_leaf_if_triangle_count_le = 16;

/** @brief Query status flags. */
constexpr u32 status_ok = 0;
constexpr u32 status_stack_overflow = 1;

struct stack_entry {
    real min_range;
    i32 ptr;
};

/** @brief Insertion sort into descending `min_range`, so the nearest child is popped first (stable, as the CPU's). */
WT_GPU_FN void sort_descending(WT_GPU_THREAD stack_entry* stack, int size) {
    for (int i=1; i<size; ++i) {
        const stack_entry p = stack[i];
        int j;
        for (j=i-1; j>=0 && p.min_range>stack[j].min_range; --j)
            stack[j+1] = stack[j];
        stack[j+1] = p;
    }
}


/*
 * Rays
 */

struct ray_hit {
    real dist;
    real baryx, baryy;  // weights of the triangle's first two vertices
    u32 tri;
    u32 hit;
    u32 front_face;
    u32 status;
};

constexpr int ray_stack_size = 64;

namespace detail {

template <bool shadow>
WT_GPU_FN bool gather_tris_ray(bvh_view bvh, const WT_GPU_THREAD ray& r,
                               u32 t0, u32 count, range rng,
                               WT_GPU_THREAD ray_hit& best) {
    bool found = false;
    for (u32 t=0; t<count; ++t) {
        const u32 tuid = t0+t;
        const bvh_tri tri = bvh.tris[tuid];
        if constexpr (shadow) {
            if (test_ray_tri_wide(r, tri.a, tri.b, tri.c, rng)) {
                best.dist = rng.min;
                return true;
            }
        } else {
            const auto h = intersect_ray_tri(r, tri.a, tri.b, tri.c, rng);
            if (h.hit && h.dist<best.dist) {
                best.dist = h.dist;
                best.baryx = h.baryx;
                best.baryy = h.baryy;
                best.tri = tuid;
                best.front_face = dot(tri.n, r.d)<=real(0) ? 1u : 0u;
                found = true;
            }
        }
    }
    return found;
}

/** @brief `traverse<shadow>(bvh8w_t*, ray_t, ...)` of src/ads/bvh8w.cpp. */
template <bool shadow>
WT_GPU_FN bool traverse_ray(bvh_view bvh, const WT_GPU_THREAD ray& r, range rng,
                            WT_GPU_THREAD ray_hit& best) {
    stack_entry stack[ray_stack_size];
    int s = 1;
    stack[0] = { real(0), bvh_root_ptr };

    while (s>0) {
        const i32 ptr = stack[s-1].ptr;
        if (bvh_is_leaf(ptr)) {
            const bvh_leaf leaf = bvh.leaves[bvh_leaf_index(ptr)];
            const bool found = gather_tris_ray<shadow>(bvh, r, leaf.tris_ptr, leaf.count, rng, best);
            --s;
            if (found) {
                if constexpr (shadow) return true;
                else while (s>0 && stack[s-1].min_range>=best.dist) --s;
            }
            continue;
        }

        const WT_GPU_GLOBAL bvh_node& n = bvh.nodes[bvh_node_index(ptr)];
        --s;

        if (n.tris_count<=bvh_treat_node_as_leaf_if_triangle_count_le) {
            const bool found = gather_tris_ray<shadow>(bvh, r, n.tris_start, n.tris_count, rng, best);
            if (found) {
                if constexpr (shadow) return true;
                else while (s>0 && stack[s-1].min_range>=best.dist) --s;
            }
            continue;
        }

        const int begin = s;
        for (int i=0;i<bvh_width;++i) {
            if (n.child[i]==0) continue;
            const auto h = intersect_ray_aabb_fast(r, bvh_child_min(n,i), bvh_child_max(n,i),
                                                   range{ real(0), best.dist });
            if (!h.hit) continue;
            if (s==ray_stack_size) {
                best.status |= status_stack_overflow;
                return false;
            }
            stack[s++] = { h.tmin, n.child[i] };
        }
        sort_descending(&stack[begin], s-begin);
    }

    return best.dist<inf();
}

}

/** @brief Closest hit along `r` within `rng` (`bvh8w_t::intersect(ray_t, range)`). */
WT_GPU_FN ray_hit intersect_ray(bvh_view bvh, ray r, range rng) {
    ray_hit best = { inf(), 0, 0, 0, 0, 0, status_ok };
    detail::traverse_ray<false>(bvh, r, rng, best);
    // ray_work_to_intersection_record
    best.hit = isfinite(best.dist) && best.dist<=rng.max ? 1u : 0u;
    return best;
}

/** @brief Whether anything blocks `r` within `rng` (`bvh8w_t::shadow(ray_t, range)`). */
WT_GPU_FN bool ray_occluded(bvh_view bvh, ray r, range rng, WT_GPU_THREAD u32& status) {
    ray_hit best = { inf(), 0, 0, 0, 0, 0, status_ok };
    const bool occluded = detail::traverse_ray<true>(bvh, r, rng, best);
    status |= best.status;
    return occluded;
}


/*
 * Cones and balls: queries returning sets of triangles
 */

/**
 * @brief Collects a query's triangles into a fixed-capacity buffer, counting all
 *        of them: when `count>capacity`, the caller re-runs the query with more room.
 */
struct tri_sink {
    WT_GPU_GLOBAL u32* out;
    u32 capacity;
    u32 count;

    WT_GPU_FN void push(u32 tuid) {
        if (count<capacity)
            out[count] = tuid;
        ++count;
    }
};

struct cone_hit {
    real dist;          // distance to the closest hit; inf if none
    u32 front_face;     // of the closest hit
    u32 tri_count;      // triangles found (may exceed the sink's capacity)
    u32 status;
};

constexpr int cone_stack_size = 128;

namespace detail {

/** @brief `intersection_record_work_t::search_range` of wt/ads/traversal_common.hpp. */
WT_GPU_FN range cone_search_range(const WT_GPU_THREAD cone& cn, range search, real intr_dist, real z_scale) {
    const real dist = max(search.min, intr_dist);
    const real z_dist = cn.axes(dist).x * z_scale;
    return range{ search.min, min(search.max, dist+z_dist) }.intersect(range::positive());
}

WT_GPU_FN bool gather_tris_cone(bvh_view bvh, const WT_GPU_THREAD cone& cn,
                                u32 t0, u32 count, range rng,
                                WT_GPU_THREAD cone_hit& rec, WT_GPU_THREAD tri_sink& sink) {
    bool found = false;
    for (u32 t=0; t<count; ++t) {
        const u32 tuid = t0+t;
        const bvh_tri tri = bvh.tris[tuid];

        const bool front_face = dot(tri.n, -cn.r.d)>real(0);
        const auto h = intersect_cone_tri(cn, tri.a, tri.b, tri.c, tri.n, rng);
        if (!h.hit || h.dist>rng.max)   // the latter can happen due to numerics
            continue;
        if (h.dist<rec.dist) {
            rec.dist = h.dist;
            rec.front_face = front_face ? 1u : 0u;
        }
        found = true;
        sink.push(tuid);
    }
    return found;
}

/**
 * @brief `traverse<shadow=false>(bvh8w_t*, elliptic_cone_t, ...)` of src/ads/bvh8w.cpp.
 *        (Cone shadow queries are not ported: the renderer only issues ray shadow queries.)
 */
WT_GPU_FN void traverse_cone(bvh_view bvh, const WT_GPU_THREAD cone& cn,
                             range search, real z_scale,
                             WT_GPU_THREAD cone_hit& rec, WT_GPU_THREAD tri_sink& sink) {
    range rng = cone_search_range(cn, search, rec.dist, z_scale);

    stack_entry stack[cone_stack_size];
    int s = 1;
    stack[0] = { real(0), bvh_root_ptr };

    while (s>0) {
        const i32 ptr = stack[s-1].ptr;
        if (bvh_is_leaf(ptr)) {
            const bvh_leaf leaf = bvh.leaves[bvh_leaf_index(ptr)];
            const bool found = gather_tris_cone(bvh, cn, leaf.tris_ptr, leaf.count, rng, rec, sink);
            --s;
            if (found) {
                // unwind irrelevant nodes
                rng = cone_search_range(cn, search, rec.dist, z_scale);
                while (s>0 && stack[s-1].min_range>=rng.max) --s;
            }
            continue;
        }

        const WT_GPU_GLOBAL bvh_node& n = bvh.nodes[bvh_node_index(ptr)];
        --s;

        const int begin = s;
        for (int i=0;i<bvh_width;++i) {
            if (n.child[i]==0) continue;
            const auto h = intersect_cone_aabb_conservative(cn, bvh_child_min(n,i), bvh_child_max(n,i), rng);
            if (!h.hit || h.tmin>=rng.max) continue;
            if (s==cone_stack_size) {
                rec.status |= status_stack_overflow;
                return;
            }
            stack[s++] = { h.tmin, n.child[i] };
        }
        sort_descending(&stack[begin], s-begin);
    }
}

}

/**
 * @brief Cone query (`bvh8w_t::intersect(elliptic_cone_t, range, opts)`): the
 *        triangles hit within `z_scale` cross-section major axes past the closest
 *        hit, in traversal order.
 */
WT_GPU_FN cone_hit intersect_cone(bvh_view bvh, cone cn, range rng, real z_scale,
                                  WT_GPU_THREAD tri_sink& sink) {
    cone_hit rec = { inf(), 0, 0, status_ok };
    detail::traverse_cone(bvh, cn, rng, z_scale, rec, sink);
    rec.tri_count = sink.count;
    return rec;
}

constexpr int ball_stack_size = 128;

namespace detail {

WT_GPU_FN void gather_tris_ball(bvh_view bvh, const WT_GPU_THREAD ball& bl,
                                u32 t0, u32 count, WT_GPU_THREAD tri_sink& sink) {
    for (u32 t=0; t<count; ++t) {
        const bvh_tri tri = bvh.tris[t0+t];
        if (test_ball_tri(bl, tri.a, tri.b, tri.c, tri.n))
            sink.push(t0+t);
    }
}

}

/** @brief Triangles overlapping the ball (`bvh8w_t::intersect(ball_t, opts)`), in traversal order. */
WT_GPU_FN u32 intersect_ball(bvh_view bvh, ball bl, WT_GPU_THREAD tri_sink& sink) {
    i32 stack[ball_stack_size];
    int s = 1;
    stack[0] = bvh_root_ptr;

    while (s>0) {
        const i32 ptr = stack[s-1];
        if (bvh_is_leaf(ptr)) {
            const bvh_leaf leaf = bvh.leaves[bvh_leaf_index(ptr)];
            detail::gather_tris_ball(bvh, bl, leaf.tris_ptr, leaf.count, sink);
            --s;
            continue;
        }

        const WT_GPU_GLOBAL bvh_node& n = bvh.nodes[bvh_node_index(ptr)];
        --s;

        if (n.tris_count<=bvh_treat_node_as_leaf_if_triangle_count_le) {
            detail::gather_tris_ball(bvh, bl, n.tris_start, n.tris_count, sink);
            continue;
        }

        for (int i=0;i<bvh_width;++i) {
            if (n.child[i]==0) continue;
            if (!test_ball_aabb(bl, bvh_child_min(n,i), bvh_child_max(n,i))) continue;
            if (s==ball_stack_size)
                return status_stack_overflow;
            stack[s++] = n.child[i];
        }
    }

    return status_ok;
}

}
