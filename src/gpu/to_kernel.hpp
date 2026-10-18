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
 * Conversions from wave_tracer types to the GPU kernel types (lengths in metres).
 */

#include <wt/gpu/kernel/shapes.hpp>

#include <wt/math/common.hpp>
#include <wt/math/frame.hpp>
#include <wt/math/range.hpp>
#include <wt/math/shapes/ball.hpp>
#include <wt/math/shapes/elliptic_cone.hpp>
#include <wt/math/shapes/ray.hpp>

namespace wt::gpu {

inline kernel::vec3 to_kernel(const vec3_t& v) noexcept {
    using kernel::real;
    return { real(v.x), real(v.y), real(v.z) };
}
inline kernel::vec3 to_kernel(const dir3_t& v) noexcept {
    return to_kernel(vec3_t{ v });
}
inline kernel::vec3 to_kernel(const pqvec3_t& p) noexcept {
    return to_kernel(vec3_t{ u::to_m(p) });
}
inline kernel::range to_kernel(const pqrange_t<>& r) noexcept {
    using kernel::real;
    return { real(u::to_m(r.min)), real(u::to_m(r.max)) };
}
inline kernel::ray to_kernel(const ray_t& r) noexcept {
    return { to_kernel(r.o), to_kernel(r.d), to_kernel(r.invd) };
}
inline kernel::frame to_kernel(const frame_t& f) noexcept {
    return { to_kernel(f.t), to_kernel(f.b), to_kernel(f.n) };
}
inline kernel::cone to_kernel(const elliptic_cone_t& c) noexcept {
    using kernel::real;
    return {
        .r = to_kernel(c.ray()),
        .x = to_kernel(c.x()),
        .y = to_kernel(c.y()),
        .x0 = real(u::to_m(c.x0())),
        .tan_alpha = real(c.get_tan_alpha()),
        .e = real(c.get_e()),
        .one_over_e = real(c.get_one_over_e()),
        .z_apex = real(u::to_m(c.get_z_apex())),
    };
}
inline kernel::ball to_kernel(const ball_t& b) noexcept {
    using kernel::real;
    return { to_kernel(b.centre), real(u::to_m(b.radius)) };
}

}
