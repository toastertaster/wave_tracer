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
#include <wt/remote_sensing/satellite_placement.hpp>
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
 *        `near_clip`. Placed by altitude, rays start `satellite_placement_t::default_near_clip_margin` short of the
 *        target by default (see `satellite_placement_t`, which it shares with the `telescope` sensor).
 */
struct satellite_frame_t {
    static std::shared_ptr<sensor::sensor_t> load(std::string id,
                                                  scene::loader::loader_t* loader,
                                                  const scene::loader::node_t& node,
                                                  const wt::wt_context_t &context);
};

}
