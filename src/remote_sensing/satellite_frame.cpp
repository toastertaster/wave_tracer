/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#include <format>
#include <optional>

#include <wt/remote_sensing/satellite_frame.hpp>

#include <wt/sensor/sensor/perspective.hpp>
#include <wt/math/transform/transform.hpp>
#include <wt/math/transform/transform_loader.hpp>
#include <wt/scene/loader/node_readers.hpp>
#include <wt/util/logger/logger.hpp>

using namespace wt;
using namespace wt::remote_sensing;

namespace {

template <bool polarimetric>
std::shared_ptr<sensor::sensor_t> load_satellite_frame(std::string id,
                                                       scene::loader::loader_t* loader,
                                                       const scene::loader::node_t& node,
                                                       const wt::wt_context_t &context) {
    using perspective_t = sensor::perspective_t<polarimetric>;

    std::optional<length_t> focal_length, pixel_pitch, altitude;
    angle_t view_zenith = 0*u::ang::rad, view_azimuth = 0*u::ang::rad;
    pqvec3_t target = pqvec3_t::zero();
    length_t earth_radius = satellite_frame_t::mean_earth_radius;
    std::optional<transform_t> to_world;

    std::uint32_t samples_per_element = 0;
    std::optional<typename perspective_t::film_t> film;
    bool ray_trace = false;
    f_t phase_space_extent_scale = 1;
    std::optional<angle_t> requested_alpha;
    std::optional<length_t> near_clip;

    for (auto& item : node.children_view()) {
    try {
        transform_t transform;
        if (item.name() == "film")
            film.emplace(perspective_t::film_t::load(loader, item, perspective_t::beam_source_spatial_stddev, context));
        else if (scene::loader::load_transform(item, "to_world", transform, loader))
            to_world = transform;
        else if (!scene::loader::read_attribute(item, "focal_length", focal_length) &&
                 !scene::loader::read_attribute(item, "pixel_pitch", pixel_pitch) &&
                 !scene::loader::read_attribute(item, "altitude", altitude) &&
                 !scene::loader::read_attribute(item, "view_zenith", view_zenith) &&
                 !scene::loader::read_attribute(item, "view_azimuth", view_azimuth) &&
                 !scene::loader::read_attribute(item, "target", target) &&
                 !scene::loader::read_attribute(item, "earth_radius", earth_radius) &&
                 !scene::loader::read_attribute(item, "samples", samples_per_element) &&
                 !scene::loader::read_attribute(item, "alpha", requested_alpha) &&
                 !scene::loader::read_attribute(item, "ray_trace_only", ray_trace) &&
                 !scene::loader::read_attribute(item, "near_clip", near_clip) &&
                 !scene::loader::read_attribute(item, "phase_space_extent_scale", phase_space_extent_scale))
            logger::cwarn()
                << loader->node_description(item)
                << "(satellite_frame sensor loader) unqueried node type " << item.name() << " (\"" << item["name"] << "\")" << '\n';
    } catch(const std::format_error& exp) {
        throw scene_loading_exception_t("(satellite_frame sensor loader) " + std::string{ exp.what() }, item);
    }
    }

    if (!film)
        throw scene_loading_exception_t("(satellite_frame sensor loader) film must be provided", node);
    if (!focal_length || *focal_length<=0*u::m)
        throw scene_loading_exception_t("(satellite_frame sensor loader) a positive 'focal_length' must be provided", node);
    if (!pixel_pitch || *pixel_pitch<=0*u::m)
        throw scene_loading_exception_t("(satellite_frame sensor loader) a positive 'pixel_pitch' must be provided", node);
    if (!!altitude == !!to_world)
        throw scene_loading_exception_t("(satellite_frame sensor loader) exactly one of 'altitude' and 'to_world' must be provided", node);
    if (altitude && *altitude<=0*u::m)
        throw scene_loading_exception_t("(satellite_frame sensor loader) 'altitude' must be positive", node);
    if (view_zenith<0*u::ang::rad || view_zenith>=m::pi/2*u::ang::rad)
        throw scene_loading_exception_t("(satellite_frame sensor loader) 'view_zenith' must be in [0°,90°)", node);
    if (requested_alpha && *requested_alpha<=zero)
        throw scene_loading_exception_t("(satellite_frame sensor loader) 'alpha' must be positive", node);
    if (near_clip && *near_clip<0*u::m)
        throw scene_loading_exception_t("(satellite_frame sensor loader) 'near_clip' must be non-negative", node);

    if (altitude) {
        // in double precision: the camera is hundreds of kilometres away
        const double H = u::to_m(*altitude), R = u::to_m(earth_radius);
        const double theta = view_zenith.numerical_value_in(u::ang::rad);
        const double phi = view_azimuth.numerical_value_in(u::ang::rad);
        const double s = satellite_frame_t::slant_range(H, theta, R);
        // towards the camera, in the scene frame (x east, y up, z south)
        const double dx = std::sin(theta)*std::sin(phi), dy = std::cos(theta), dz = -std::sin(theta)*std::cos(phi);
        const auto t = u::to_m(target);
        const auto origin = pqvec3_t{ vec3_t{ t.x + s*dx, t.y + s*dy, t.z + s*dz } * u::m };
        // north up, at nadir
        to_world = transform_t::lookat(origin, target, vec3_t{ 0,0,-1 });
        // start rays near the scene (see perspective_t::near_clip)
        if (!near_clip)
            near_clip = m::max<double>(0, s - u::to_m(satellite_frame_t::default_near_clip_margin)) * u::m;

        logger::cout(verbosity_e::info)
            << loader->node_description(node)
            << std::format("(satellite_frame sensor) <{}> at {:.3f} km altitude, view zenith {:.3f}°, azimuth {:.3f}°: slant range {:.3f} km, rays start at {:.3f} km",
                           id, H/1e3, theta*180/m::pi, phi*180/m::pi, s/1e3, u::to_m(*near_clip)/1e3)
            << '\n';
    }

    // exact pixel geometry: the film's N pixels span N pixel pitches on the focal plane (a perspective sensor's
    // fov-based transform maps the FOV onto N-1 pixels)
    const auto dims = vec2_t{ film->dimensions() };
    const auto pitch_over_f = f_t(u::to_m(*pixel_pitch) / u::to_m(*focal_length));
    const angle_t fov = 2*m::atan(dims.y*pitch_over_f/2);
    const auto viewport =
        transform_t::scale({ dims.x/2, dims.y/2, 1 }) *
        transform_t::translate(pqvec3_t{ vec3_t{ 1,1,0 } * u::m }) *     // unitless, as transform_t::viewport's
        transform_t::scale({ -1,-1,1 });
    const auto film_to_camera = (
        viewport *
        transform_t::perspective(fov, dims.x/dims.y, u::to_m(perspective_t::image_plane_z))
    ).inverse();

    return std::make_shared<perspective_t>(
        context,
        std::move(id),
        *to_world,
        film_to_camera,
        fov,
        std::move(film).value(),
        samples_per_element,
        ray_trace || context.renderer_force_ray_tracing,
        nullptr,
        requested_alpha ? m::tan(*requested_alpha) : std::optional<f_t>{},
        phase_space_extent_scale,
        near_clip.value_or(0*u::m)
    );
}

}


std::shared_ptr<sensor::sensor_t> satellite_frame_t::load(std::string id,
                                                          scene::loader::loader_t* loader,
                                                          const scene::loader::node_t& node,
                                                          const wt::wt_context_t &context) {
    if (stob_strict(node["polarimetric"], false))
        return load_satellite_frame<true>(std::move(id), loader, node, context);
    return load_satellite_frame<false>(std::move(id), loader, node, context);
}
