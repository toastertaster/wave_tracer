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

    std::optional<length_t> focal_length, pixel_pitch;
    satellite_placement_t placement;

    std::uint32_t samples_per_element = 0;
    std::optional<typename perspective_t::film_t> film;
    bool ray_trace = false;
    f_t phase_space_extent_scale = 1;
    std::optional<angle_t> requested_alpha;

    for (auto& item : node.children_view()) {
    try {
        if (item.name() == "film")
            film.emplace(perspective_t::film_t::load(loader, item, perspective_t::beam_source_spatial_stddev, context));
        else if (!placement.read(item, loader) &&
                 !scene::loader::read_attribute(item, "focal_length", focal_length) &&
                 !scene::loader::read_attribute(item, "pixel_pitch", pixel_pitch) &&
                 !scene::loader::read_attribute(item, "samples", samples_per_element) &&
                 !scene::loader::read_attribute(item, "alpha", requested_alpha) &&
                 !scene::loader::read_attribute(item, "ray_trace_only", ray_trace) &&
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
    if (requested_alpha && *requested_alpha<=zero)
        throw scene_loading_exception_t("(satellite_frame sensor loader) 'alpha' must be positive", node);

    const auto placed = placement.resolve("satellite_frame", id, loader, node);
    // north up, at nadir
    const auto to_world = placed.by_altitude ?
        transform_t::lookat(pqvec3_t{ vec3_t{ placed.by_altitude->position } * u::m },
                            pqvec3_t{ vec3_t{ placed.by_altitude->target } * u::m },
                            vec3_t{ 0,0,-1 }) :
        static_cast<transform_t>(*placed.to_world);

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
        to_world,
        film_to_camera,
        fov,
        std::move(film).value(),
        samples_per_element,
        ray_trace || context.renderer_force_ray_tracing,
        nullptr,
        requested_alpha ? m::tan(*requested_alpha) : std::optional<f_t>{},
        phase_space_extent_scale,
        placed.near_clip
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
