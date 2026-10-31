/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include "bvh.hpp"

/*
 * Batched acceleration-structure queries, as passed between the host and the
 * backends' query kernels (see wt/gpu/backend.hpp).
 */

namespace wt { namespace gpu { namespace compute_kernel {

struct ray_query {
    compute_kernel::ray ray;
    compute_kernel::range range;
};

struct cone_query {
    compute_kernel::cone cone;
    compute_kernel::range range;
    /** @brief Triangles are collected this many cross-section major axes past the closest hit. */
    real z_scale;
};

/** @brief Result of a ball query. */
struct ball_hit {
    u32 tri_count;      // may exceed the capacity the query was run with
    u32 status;
};

} } }
