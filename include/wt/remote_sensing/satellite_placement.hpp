/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include <cmath>
#include <optional>
#include <string>
#include <string_view>

#include <wt/math/common.hpp>
#include <wt/math/transform/transform.hpp>

namespace wt::scene::loader {
class loader_t;
class node_t;
}

namespace wt::remote_sensing {

/**
 * @brief Where a satellite's camera is: the placement shared by the remote sensing sensors (`satellite_frame`,
 *        `telescope`).
 *
 *        Scenes use a local frame at the target's ground: x east, y up, z south (so that north is -z).
 *        The camera is at `altitude` above a spherical Earth of radius `earth_radius`, seen from `target` at view
 *        zenith angle `view_zenith` and view azimuth `view_azimuth` (clockwise from north, towards the camera).
 *        Alternatively, a `to_world` transform places it.
 *
 *        Placed by altitude, rays start `default_near_clip_margin` short of the target, unless `near_clip` is given
 *        (see `perspective_t::near_clip`).
 */
struct satellite_placement_t {
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

    std::optional<length_t> altitude;
    angle_t view_zenith = 0*u::ang::rad, view_azimuth = 0*u::ang::rad;
    pqvec3_t target = pqvec3_t::zero();
    length_t earth_radius = mean_earth_radius;
    std::optional<transform_d_t> to_world;
    std::optional<length_t> near_clip;

    /**
     * @brief Reads `item` if it is one of the placement's attributes: `altitude`, `view_zenith`, `view_azimuth`,
     *        `target`, `earth_radius`, `near_clip` or the `to_world` transform. Returns TRUE if it was.
     */
    bool read(const scene::loader::node_t& item, scene::loader::loader_t* loader);

    /**
     * @brief The placement, resolved.
     */
    struct resolved_t {
        /** @brief Placed by altitude: the camera's position and the target's, in world coordinates (metres). */
        struct by_altitude_t {
            glm::dvec3 position;
            glm::dvec3 target;
        };
        std::optional<by_altitude_t> by_altitude;
        /** @brief Placed by `to_world`: the user's transform. */
        std::optional<transform_d_t> to_world;
        /** @brief Rays start this far from the camera. */
        length_t near_clip;
    };

    /**
     * @brief Checks the attributes read and places the camera (logging where, when placed by altitude).
     *        Throws `scene_loading_exception_t` naming `sensor_type` on invalid attributes.
     */
    [[nodiscard]] resolved_t resolve(std::string_view sensor_type,
                                     const std::string& id,
                                     scene::loader::loader_t* loader,
                                     const scene::loader::node_t& node) const;
};

}
