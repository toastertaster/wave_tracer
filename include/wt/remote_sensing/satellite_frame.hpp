/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include <memory>
#include <string>

#include <wt/sensor/sensor.hpp>
#include <wt/math/common.hpp>

namespace wt::remote_sensing {

/**
 * @brief A satellite's frame (staring) camera: a pinhole camera of a given focal length and pixel pitch, placed by
 *        its altitude and the direction it is seen from a target on the ground.
 *
 *        Scenes use a local frame at the target's ground: x east, y up, z south (so that north is -z).
 *        The camera is at `altitude` above a spherical Earth of radius `earth_radius`, seen from `target` at view
 *        zenith angle `view_zenith` and view azimuth `view_azimuth` (clockwise from north, towards the camera).
 *        It looks at `target`, with north up in its images (east to the right) at nadir.
 *        Alternatively, a `to_world` transform places it as a perspective sensor's.
 *
 *        The camera is a `perspective` sensor with exact pixel geometry: pixel (i,j) spans the angles of a pixel
 *        pitch on the focal plane, about the optical axis through the film's centre. Its field of view (vertical)
 *        is 2 atan(height · pitch / (2 · focal length)).
 *
 *        Loaded from `<sensor type="satellite_frame">`, with `focal_length`, `pixel_pitch`, a film, and either
 *        `altitude` (and optionally `view_zenith`, `view_azimuth`, `target`, `earth_radius`) or `to_world`.
 *        Also accepts the perspective sensor's `samples`, `ray_trace_only`, `alpha`, `phase_space_extent_scale` and
 *        `near_clip`. Placed by altitude, rays start `default_near_clip_margin` short of the target by default.
 */
struct satellite_frame_t {
    /**
     * @brief Mean Earth radius (IUGG).
     */
    static constexpr auto mean_earth_radius = 6371008.8 * u::m;
    /**
     * @brief Placed by altitude, rays start this far short of the target (see `perspective_t::near_clip`), unless
     *        `near_clip` is given: geometry farther than this from the target's ground, along the line of sight, is
     *        not seen (give `near_clip` for taller scenes). Nearer is more precise: from 500 km at 30° off nadir,
     *        starting 20 km short still leaves dark samples (self-shadowing) on ~0.5% of terrain pixels, from 1 km
     *        as few as at nadir.
     */
    static constexpr auto default_near_clip_margin = 2 * u::km;

    /**
     * @brief Distance from a target to a camera at altitude `altitude` above a spherical Earth of radius `R`, seen
     *        at view zenith angle `view_zenith` from the target.
     */
    [[nodiscard]] static double slant_range(double altitude, double view_zenith, double R) noexcept {
        const auto c = std::cos(view_zenith);
        return -R*c + std::sqrt(R*R*c*c + 2*R*altitude + altitude*altitude);
    }

    static std::shared_ptr<sensor::sensor_t> load(std::string id,
                                                  scene::loader::loader_t* loader,
                                                  const scene::loader::node_t& node,
                                                  const wt::wt_context_t &context);
};

}
