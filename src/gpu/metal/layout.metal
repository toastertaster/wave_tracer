/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

/*
 * Metal backend: the struct layout check (see layout.hpp).
 */

#include "layout.hpp"

kernel void struct_layouts(device wt::gpu::compute_kernel::u32* out [[buffer(0)]],
                           uint i [[thread_position_in_grid]]) {
    if (i==0)
        wt::gpu::metal_backend::write_layouts(out);
}
