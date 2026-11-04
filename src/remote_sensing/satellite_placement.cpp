/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#include <format>
#include <string>

#include <wt/remote_sensing/satellite_placement.hpp>

#include <wt/scene/loader/loader.hpp>
#include <wt/scene/loader/node.hpp>
#include <wt/scene/loader/node_readers.hpp>
#include <wt/math/transform/transform_loader.hpp>
#include <wt/util/logger/logger.hpp>

using namespace wt;
using namespace wt::remote_sensing;


bool satellite_placement_t::read(const scene::loader::node_t& item, scene::loader::loader_t* loader) {
    transform_d_t transform;
    if (scene::loader::load_transform(item, "to_world", transform, loader)) {
        if (to_world)
            throw std::format_error("Node \"to_world\" already provided");
        to_world = transform;
        return true;
    }
    return scene::loader::read_attribute(item, "altitude", altitude) ||
           scene::loader::read_attribute(item, "view_zenith", view_zenith) ||
           scene::loader::read_attribute(item, "view_azimuth", view_azimuth) ||
           scene::loader::read_attribute(item, "target", target) ||
           scene::loader::read_attribute(item, "earth_radius", earth_radius) ||
           scene::loader::read_attribute(item, "near_clip", near_clip);
}

satellite_placement_t::resolved_t satellite_placement_t::resolve(std::string_view sensor_type,
                                                                 const std::string& id,
                                                                 scene::loader::loader_t* loader,
                                                                 const scene::loader::node_t& node) const {
    const auto error = [&](std::string_view what) {
        return scene_loading_exception_t(std::format("({} sensor loader) {}", sensor_type, what), node);
    };

    if (!!altitude == !!to_world)
        throw error("exactly one of 'altitude' and 'to_world' must be provided");
    if (altitude && *altitude<=0*u::m)
        throw error("'altitude' must be positive");
    if (view_zenith<0*u::ang::rad || view_zenith>=m::pi/2*u::ang::rad)
        throw error("'view_zenith' must be in [0°,90°)");
    if (near_clip && *near_clip<0*u::m)
        throw error("'near_clip' must be non-negative");

    if (to_world)
        return { .to_world = to_world, .near_clip = near_clip.value_or(0*u::m) };

    // in double precision: the camera is hundreds of kilometres away
    const double H = u::to_m(*altitude), R = u::to_m(earth_radius);
    const double theta = view_zenith.numerical_value_in(u::ang::rad);
    const double phi = view_azimuth.numerical_value_in(u::ang::rad);
    const double s = slant_range(H, theta, R);
    // towards the camera, in the scene frame (x east, y up, z south)
    const auto towards = glm::dvec3{ std::sin(theta)*std::sin(phi), std::cos(theta), -std::sin(theta)*std::cos(phi) };
    const auto t = glm::dvec3{ u::to_m(target) };
    // start rays near the scene (see perspective_t::near_clip)
    const auto clip = near_clip.value_or(m::max<double>(0, s - u::to_m(default_near_clip_margin)) * u::m);

    logger::cout(verbosity_e::info)
        << loader->node_description(node)
        << std::format("({} sensor) <{}> at {:.3f} km altitude, view zenith {:.3f}°, azimuth {:.3f}°: slant range {:.3f} km, rays start at {:.3f} km",
                       sensor_type, id, H/1e3, theta*180/m::pi, phi*180/m::pi, s/1e3, u::to_m(clip)/1e3)
        << '\n';

    return {
        .by_altitude = resolved_t::by_altitude_t{ .position = t + s*towards, .target = t },
        .near_clip = clip,
    };
}
