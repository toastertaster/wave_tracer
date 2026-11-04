/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#include <cmath>
#include <charconv>
#include <cstdlib>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <wt/remote_sensing/telescope.hpp>
#include <wt/remote_sensing/satellite_placement.hpp>
#include "prescription.hpp"

#include <wt/scene/element/attributes.hpp>
#include <wt/scene/loader/node_readers.hpp>
#include <wt/math/transform/transform.hpp>
#include <wt/math/transform/transform_loader.hpp>
#include <wt/util/format/parse.hpp>
#include <wt/util/logger/logger.hpp>

using namespace wt;
using namespace wt::remote_sensing;


namespace {

inline dir3_t to_dir(const glm::dvec3& v) noexcept {
    return dir3_t{ vec3_t{ glm::normalize(v) } };
}

// the s-p frame of direction w at a surface of unit normal n, as wave_tracer's surfaces' (intersection_surface_t::
// sp_frame): s perpendicular to the plane of incidence, the same for w towards the surface and away from it
frame_t sp_frame(const dir3_t& w, const dir3_t& n) noexcept {
    const auto crs = m::cross(w, n);
    const auto l2 = m::length2(crs);
    auto s = l2<f_t(1e-14) ? frame_t::build_orthogonal_frame(n).t : dir3_t{ crs/m::sqrt(l2) };
    const bool inwards = m::dot(w,n)<0;
    if (inwards)
        s = -s;
    const auto p = dir3_t{ m::cross(s, w) };
    return frame_t{ .t = s, .b = inwards ? -p : p, .n = w };
}

}


template <bool polarimetric>
frame_t telescope_t<polarimetric>::detector_frame(const glm::dvec3& d) const noexcept {
    const auto x = glm::dvec3{ fp.to_telescope[0] }, y = glm::dvec3{ fp.to_telescope[1] };
    const auto t = glm::normalize(x - glm::dot(x,d)*d);
    const auto b = glm::normalize(y - glm::dot(y,d)*d - glm::dot(y,t)*t);
    return frame_t{ .t = to_dir(t), .b = to_dir(b), .n = to_dir(d) };
}

template <bool polarimetric>
sensor_sample_t telescope_t<polarimetric>::sample(sampler::sampler_t& sampler,
                                                  const vec3u32_t& sensor_element,
                                                  const wavenumber_t k) const noexcept {
    const auto film_pos = vec2u32_t{ sensor_element };
    const auto offset = sampler.r2();
    const auto p = focal_plane_point(vec2_t{ film_pos } + offset);

    // a direction towards the optics, from the mixture of the cone of passing directions and the cone bounding all
    // that can pass, traced backwards
    const auto tight = passing_cone(vec2_t{ film_pos } + offset);
    const auto bound = optics->aim(p);
    const auto u = sampler.r2();
    const auto d = sampler.r()<aim_tight_fraction ? tight.sample(u.x, u.y) : bound.sample(u.x, u.y);
    const auto pdf =
        (tight.contains(d) ? aim_tight_fraction / tight.solid_angle() : 0.) +
        (bound.contains(d) ? (1-aim_tight_fraction) / bound.solid_angle() : 0.);
    const auto traced = optics->trace_from_focal_plane({ p,d });

    // projected solid angle over its sampling density, relative to the film centre's
    const auto weight = traced.passed && pdf>0 ?
        std::abs(glm::dot(d, focal_plane_normal())) / pdf / props.centre.projected_solid_angle :
        0.;

    // towards the scene, in world coordinates (in double precision: the telescope is far away)
    const auto exit = traced.passed ? traced.exit : optics::ray_t{ p, props.centre.line_of_sight };
    const auto wd = glm::normalize(place.rotation * exit.d);
    const auto wo = place.position + place.rotation * exit.o;
    const auto dir = dir3_t{ vec3_t{ wd } };

    // the polarization: the mirrors' reflections, composed as wave_tracer's surfaces compose theirs (in s-p frames),
    // from the detector's frame (Q along the film's x) to the frame of the light arriving from the scene
    auto M = mueller_operator_t::identity();
    auto F = detector_frame(d);
    for (std::uint32_t i=0; i<traced.reflection_count && traced.passed; ++i) {
        const auto& r = traced.reflections[i];
        const auto n = to_dir(r.normal);
        const auto c = optics->elements()[r.element].coating->reflection(k, glm::dot(r.d_out, r.normal));
        M = compose(M, mueller_operator_t::fresnel(c.rs, c.rp), F, sp_frame(F.n, n));
        F = sp_frame(to_dir(r.d_out), n);
    }
    // ... in world coordinates
    const auto Fw = frame_t{
        .t = to_dir(place.rotation * glm::dvec3{ vec3_t{ F.t } }),
        .b = to_dir(place.rotation * glm::dvec3{ vec3_t{ F.b } }),
        .n = dir,
    };

    const QE_flux_t W = f_t(weight) * u::ang::sr * square(u::m);
    auto beam = importance_flux_beam_t{ ray_t{ pqvec3_t{ vec3_t{ wo } * u::m }, dir }, M, W, Fw, k, sourcing_geometry(k) };
    // start near the scene: the beam as propagated through vacuum to there
    if (place.near_clip>0)
        beam.transform_restart(pqvec3_t{ vec3_t{ wo + place.near_clip*wd } * u::m }, f_t(place.near_clip) * u::m);

    return sensor_sample_t{
        .sensor = this,

        .beam = beam,

        // on the focal plane
        .ppd = area_sampling_pd_t{ f_t(1 / (fp.pixel_pitch*fp.pixel_pitch)) / square(u::m) },
        .dpd = solid_angle_density_t{ f_t(pdf) / u::ang::sr },

        .element = sensor::sensor_element_sample_t{
            .element = { film_pos, 0 },
            .offset  = { offset - vec2_t{ .5,.5 }, 0 },
        },
    };
}

template <bool polarimetric>
sensor_direct_sample_t telescope_t<polarimetric>::sample_direct(sampler::sampler_t& sampler,
                                                                const pqvec3_t& wp,
                                                                const wavenumber_t k) const noexcept {
    const auto dir = dir3_t{ vec3_t{ glm::normalize(place.rotation * props.centre.line_of_sight) } };
    return sensor_direct_sample_t{
        .sensor = this,
        .beam = importance_beam_t{ ray_t{ wp, -dir }, QE_t::zero(), k, sourcing_geometry(k) },
        .dpd = solid_angle_sampling_pd_t::discrete(1),
        .element = {},
    };
}

template <bool polarimetric>
scene::element::info_t telescope_t<polarimetric>::description() const {
    using namespace scene::element;

    auto film_info = this->film().description();
    auto film_desc = std::move(film_info.attribs);
    film_desc.emplace("cls", attributes::make_string(film_info.cls));

    const auto& c = props.centre;
    const auto los = place.rotation * c.line_of_sight;

    auto info = info_for_scene_element(*this, "telescope", {
        { "position (m)",               attributes::make_vector(vec3_t{ place.position }) },
        { "line of sight",              attributes::make_vector(vec3_t{ los }) },
        { "elements",                   attributes::make_scalar(optics->elements().size()) },
        { "pixel pitch (m)",            attributes::make_scalar(fp.pixel_pitch) },
        { "focal length x (m)",         attributes::make_scalar(props.focal_length_x) },
        { "focal length y (m)",         attributes::make_scalar(props.focal_length_y) },
        { "entrance pupil diameter (m)", attributes::make_scalar(c.pupil_diameter) },
        { "obscuration",                attributes::make_scalar(c.pupil_obscuration) },
        { "projected solid angle (sr)", attributes::make_scalar(c.projected_solid_angle) },
        { "centre RMS blur (rad)",      attributes::make_scalar(c.rms_spread) },
        { "corner relative throughput", attributes::make_scalar(props.corner_relative_throughput) },
        { "film",                       attributes::make_map(std::move(film_desc)) },
    });

    // the mirrors' coatings, and their reflectance at normal incidence at the film's mean wavelength
    const auto k = this->sensitivity_spectrum().mean_wavenumber();
    info.attribs.emplace("reflectance wavelength (nm)",
                         attributes::make_scalar(double(u::to_m(wavenum_to_wavelen(k))) * 1e9));
    for (std::size_t i=0; i<optics->elements().size(); ++i) {
        const auto& e = optics->elements()[i];
        if (e.kind!=optics::element_kind_e::mirror)
            continue;
        const auto name = e.id.empty() ? std::format("#{}", i) : e.id;
        info.attribs.emplace("coating " + name, attributes::make_string(e.coating->description()));
        info.attribs.emplace("normal reflectance " + name, attributes::make_scalar(e.coating->normal_reflectance(k)));
    }
    return info;
}

template class wt::remote_sensing::telescope_t<false>;
template class wt::remote_sensing::telescope_t<true>;


namespace {

using prescription::read_length;
using prescription::read_number;
using prescription::read_length_vec2;

// a rigid transform. The scene loader reads transforms' numbers in single precision (matrices' entries, rotations'
// angles): their rotation is orthonormalized again, in double precision
glm::dmat4 rigid_matrix(const transform_d_t& t, const std::string& what, const scene::loader::node_t& node) {
    auto m = glm::dmat4{ t.matrix() };
    const auto R = glm::dmat3{ m };
    const auto E = glm::transpose(R)*R - glm::dmat3(1);
    for (int c=0; c<3; ++c)
    for (int r=0; r<3; ++r)
        if (std::abs(E[c][r])>1e-5)
            throw scene_loading_exception_t("(telescope sensor loader) '" + what + "' must be a rigid transform (rotations and translations)", node);
    if (glm::determinant(R)<0)
        throw scene_loading_exception_t("(telescope sensor loader) '" + what + "' must not reflect", node);
    const auto x = glm::normalize(R[0]);
    const auto z = glm::normalize(glm::cross(x, R[1]));
    const auto y = glm::cross(z, x);
    m[0] = glm::dvec4{ x,0 };
    m[1] = glm::dvec4{ y,0 };
    m[2] = glm::dvec4{ z,0 };
    return m;
}

optics::element_t load_element(scene::loader::loader_t* loader, const scene::loader::node_t& node,
                               const wt::wt_context_t &context) {
    optics::element_t e;
    e.id = node["id"];
    if      (node.name()=="mirror")      e.kind = optics::element_kind_e::mirror;
    else if (node.name()=="stop")        e.kind = optics::element_kind_e::stop;
    else                                 e.kind = optics::element_kind_e::obscuration;
    const auto what = std::format("<{}{}>", node.name(), e.id.empty() ? "" : " id=\"" + e.id + "\"");

    std::optional<double> radius, conic, aperture_radius, hole_radius;
    std::optional<glm::dvec2> aperture_centre;
    std::optional<transform_d_t> to_telescope;
    std::vector<std::optional<double>> aspheric;
    std::shared_ptr<const optics::coating_t> coating;

    for (auto& item : node.children_view()) {
    try {
        // aspheric coefficients: a4, a6, ... (of r^4, r^6, ...)
        const auto& name = item["name"];
        if (item.name()=="float" && name.size()>1 && name[0]=='a') {
            int power = 0;
            const auto [ptr, ec] = std::from_chars(name.data()+1, name.data()+name.size(), power);
            if (ec==std::errc{} && ptr==name.data()+name.size() && power>=4 && power%2==0) {
                const auto i = std::size_t(power-4)/2;
                if (aspheric.size()<=i) aspheric.resize(i+1);
                read_number(item, name, aspheric[i]);
                continue;
            }
        }
        transform_d_t t;
        if (scene::loader::load_transform(item, "to_telescope", t, loader)) {
            if (to_telescope)
                throw std::format_error("Node \"to_telescope\" already provided");
            to_telescope = t;
        }
        else if (item.name()=="coating") {
            if (coating)
                throw std::format_error("Node \"coating\" already provided");
            coating = optics::coating_t::load(loader, item, context);
        }
        else if (!read_length(item, "radius", radius) &&
                 !read_number(item, "conic", conic) &&
                 !read_length(item, "aperture_radius", aperture_radius) &&
                 !read_length(item, "hole_radius", hole_radius) &&
                 !read_length_vec2(item, "aperture_centre", aperture_centre))
            logger::cwarn()
                << loader->node_description(item)
                << "(telescope sensor loader) " << what << ": unqueried node type " << item.name() << " (\"" << item["name"] << "\")" << '\n';
    } catch(const std::format_error& exp) {
        throw scene_loading_exception_t("(telescope sensor loader) " + what + ": " + std::string{ exp.what() }, item);
    }
    }

    const auto error = [&](std::string_view msg) {
        return scene_loading_exception_t(std::format("(telescope sensor loader) {}: {}", what, msg), node);
    };
    if (!aperture_radius)
        throw error("'aperture_radius' must be provided");
    if (!to_telescope)
        throw error("the 'to_telescope' transform must be provided");
    if (e.kind!=optics::element_kind_e::mirror && (radius || conic || !aspheric.empty()))
        throw error("stops and obscurations are flat: no 'radius', 'conic' or aspheric terms");
    if (e.kind!=optics::element_kind_e::mirror && coating)
        throw error("only mirrors have a coating");
    if (radius && *radius==0)
        throw error("'radius' must not be 0 (omit it for a flat mirror)");

    e.shape.curvature = radius ? 1 / *radius : 0;
    e.shape.conic = conic.value_or(0);
    for (const auto& a : aspheric)
        e.shape.aspheric.push_back(a.value_or(0));
    e.aperture = optics::aperture_t{
        .radius = *aperture_radius,
        .hole_radius = hole_radius.value_or(0),
        .centre = aperture_centre.value_or(glm::dvec2{ 0,0 }),
    };
    e.coating = coating ? std::move(coating) : std::make_shared<optics::coating_t>();
    e.to_telescope = rigid_matrix(*to_telescope, "to_telescope", node);
    e.to_local = glm::inverse(e.to_telescope);
    return e;
}

// the rotation taking unit vector `from` to unit vector `to` (Rodrigues)
glm::dmat3 rotation_between(const glm::dvec3& from, const glm::dvec3& to) {
    const auto v = glm::cross(from, to);
    const auto c = glm::dot(from, to);
    if (c < -1 + 1e-12) {
        // opposite: half a turn about any perpendicular axis
        const auto helper = std::abs(from.x)<.9 ? glm::dvec3{ 1,0,0 } : glm::dvec3{ 0,1,0 };
        const auto a = glm::normalize(glm::cross(from, helper));
        return 2.*glm::outerProduct(a,a) - glm::dmat3(1);
    }
    const auto K = glm::dmat3{ 0, v.z, -v.y,   -v.z, 0, v.x,   v.y, -v.x, 0 };     // column major: K x = v × x
    return glm::dmat3(1) + K + K*K / (1+c);
}

// rays per analysed focal-plane point
constexpr std::uint32_t centre_samples = 1u<<18;
constexpr std::uint32_t field_samples = 1u<<16;
constexpr std::uint32_t aim_samples = 1u<<14;

template <bool polarimetric>
std::shared_ptr<sensor::sensor_t> load(std::string id,
                                       scene::loader::loader_t* loader,
                                       const scene::loader::node_t& node,
                                       const wt::wt_context_t &context) {
    using telescope_t = remote_sensing::telescope_t<polarimetric>;

    satellite_placement_t placement;
    std::optional<double> pixel_pitch, focal_length;
    std::optional<transform_d_t> focal_plane;
    std::vector<optics::element_t> elements;

    std::uint32_t samples_per_element = 0;
    std::optional<typename telescope_t::film_t> film;
    bool ray_trace = false;
    f_t phase_space_extent_scale = 1;
    std::optional<angle_t> requested_alpha;

    for (auto& item : node.children_view()) {
    try {
        transform_d_t t;
        if (item.name() == "film")
            film.emplace(telescope_t::film_t::load(loader, item, telescope_t::beam_source_spatial_stddev, context));
        else if (item.name()=="mirror" || item.name()=="stop" || item.name()=="obscuration")
            elements.emplace_back(load_element(loader, item, context));
        else if (scene::loader::load_transform(item, "focal_plane", t, loader)) {
            if (focal_plane)
                throw std::format_error("Node \"focal_plane\" already provided");
            focal_plane = t;
        }
        else if (!placement.read(item, loader) &&
                 !read_length(item, "pixel_pitch", pixel_pitch) &&
                 !read_length(item, "focal_length", focal_length) &&
                 !scene::loader::read_attribute(item, "samples", samples_per_element) &&
                 !scene::loader::read_attribute(item, "alpha", requested_alpha) &&
                 !scene::loader::read_attribute(item, "ray_trace_only", ray_trace) &&
                 !scene::loader::read_attribute(item, "phase_space_extent_scale", phase_space_extent_scale))
            logger::cwarn()
                << loader->node_description(item)
                << "(telescope sensor loader) unqueried node type " << item.name() << " (\"" << item["name"] << "\")" << '\n';
    } catch(const std::format_error& exp) {
        throw scene_loading_exception_t("(telescope sensor loader) " + std::string{ exp.what() }, item);
    }
    }

    const auto error = [&](std::string_view msg) {
        return scene_loading_exception_t(std::format("(telescope sensor loader) <{}>: {}", id, msg), node);
    };
    if (!film)
        throw error("film must be provided");
    if (!pixel_pitch || !(*pixel_pitch>0))
        throw error("a positive 'pixel_pitch' must be provided");
    if (focal_length && !(*focal_length>0))
        throw error("'focal_length' must be positive");
    if (!focal_plane)
        throw error("the 'focal_plane' transform (focal-plane array to telescope frame) must be provided");
    if (requested_alpha && *requested_alpha<=zero)
        throw error("'alpha' must be positive");

    std::shared_ptr<const optics::telescope_optics_t> optics;
    try {
        optics = std::make_shared<optics::telescope_optics_t>(std::move(elements));
    } catch (const std::invalid_argument& exp) {
        throw error(exp.what());
    }

    const auto fp = typename telescope_t::focal_plane_t{
        .to_telescope = rigid_matrix(*focal_plane, "focal_plane", node),
        .pixel_pitch = *pixel_pitch,
    };
    const auto dims = glm::dvec2{ film->dimensions() };
    const auto fp_point = [&](glm::dvec2 film_pos) {
        return glm::dvec3{ fp.to_telescope * glm::dvec4{ (film_pos - dims/2.) * fp.pixel_pitch, 0, 1 } };
    };
    const auto fp_normal = glm::normalize(glm::dvec3{ fp.to_telescope * glm::dvec4{ 0,0,1,0 } });

    // what the optics do: the film centre's rays
    typename telescope_t::properties_t props;
    props.centre = optics->analyse(fp_point(dims/2.), fp_normal, centre_samples);
    if (props.centre.passed_fraction==0)
        throw error("no ray from the focal plane's centre passes through the optics: check the elements' placement, apertures and order (in the order light meets them), and the focal plane's");
    if (!(props.centre.line_of_sight.z<0))
        throw error("the film centre's line of sight does not look towards the scene (-z in the telescope frame, against the light): check the elements' placement and the focal plane's");

    // focal lengths: lines of sight about the film's centre
    const auto delta = 10.;     // pixels
    const auto focal_length_along = [&](glm::dvec2 axis) {
        const auto a = optics->analyse(fp_point(dims/2. - delta*axis), fp_normal, field_samples);
        const auto b = optics->analyse(fp_point(dims/2. + delta*axis), fp_normal, field_samples);
        if (a.passed_fraction==0 || b.passed_fraction==0)
            throw error("no ray passes through the optics near the film's centre");
        const auto angle = std::atan2(glm::length(glm::cross(a.line_of_sight, b.line_of_sight)),
                                      glm::dot(a.line_of_sight, b.line_of_sight));
        return 2*delta*fp.pixel_pitch / angle;
    };
    props.focal_length_x = focal_length_along({ 1,0 });
    props.focal_length_y = focal_length_along({ 0,1 });
    const auto f = (props.focal_length_x + props.focal_length_y) / 2;
    if (focal_length && std::abs(f / *focal_length - 1) > .01)
        throw error(std::format("the optics' focal length is {:.6g} m (x {:.6g} m, y {:.6g} m), not the 'focal_length' given ({:.6g} m)",
                                f, props.focal_length_x, props.focal_length_y, *focal_length));

    // vignetting: the corners' throughput
    props.corner_relative_throughput = 1;
    for (const auto corner : { glm::dvec2{ .5,.5 }, glm::dvec2{ dims.x-.5,.5 },
                               glm::dvec2{ .5,dims.y-.5 }, glm::dvec2{ dims.x-.5,dims.y-.5 } }) {
        const auto c = optics->analyse(fp_point(corner), fp_normal, field_samples);
        props.corner_relative_throughput = std::min(props.corner_relative_throughput,
                                                    c.projected_solid_angle / props.centre.projected_solid_angle);
    }

    // where samples aim: the cones of passing directions, on a grid spanning the film (where no ray passes, the
    // bounding cone)
    constexpr auto n = telescope_t::aim_grid_size;
    std::vector<optics::cone_t> aim_grid;
    for (std::uint32_t j=0; j<n; ++j)
    for (std::uint32_t i=0; i<n; ++i) {
        const auto p = fp_point(glm::dvec2{ i,j } / double(n-1) * dims);
        const auto a = optics->analyse(p, fp_normal, aim_samples);
        aim_grid.push_back(a.passed_fraction>0 && a.passing_cone.cos_half_angle<1 ? a.passing_cone : optics->aim(p));
    }

    const auto blur_pixels = props.centre.rms_spread * f / fp.pixel_pitch;
    if (blur_pixels>10)
        logger::cwarn()
            << loader->node_description(node)
            << std::format("(telescope sensor loader) <{}>: the film centre's geometric blur is {:.3g} pixels (RMS): is the focal plane in focus?", id, blur_pixels)
            << '\n';

    // telescope frame to world
    const auto placed = placement.resolve("telescope", id, loader, node);
    typename telescope_t::placement_t place;
    place.near_clip = u::to_m(placed.near_clip);
    if (placed.by_altitude) {
        // at nadir: x east, y north, z up (towards the camera); the film centre's line of sight on the target
        const auto z = glm::normalize(placed.by_altitude->position - placed.by_altitude->target);
        const auto north = glm::dvec3{ 0,0,-1 };
        const auto y = glm::normalize(north - glm::dot(north,z)*z);
        const auto x = glm::cross(y,z);
        place.rotation = glm::dmat3{ x,y,z } * rotation_between(props.centre.line_of_sight, { 0,0,-1 });
        place.position = placed.by_altitude->position;
    } else {
        const auto m = rigid_matrix(*placed.to_world, "to_world", node);
        place.rotation = glm::dmat3{ m };
        place.position = glm::dvec3{ m[3] };
    }

    const auto tan_alpha = requested_alpha ? f_t(m::tan(*requested_alpha)) : f_t(fp.pixel_pitch / f);

    logger::cout(verbosity_e::info)
        << loader->node_description(node)
        << std::format("(telescope sensor) <{}>: focal length {:.6g} m (x {:.6g} m, y {:.6g} m), entrance pupil {:.4g} m (obscuration {:.3g}), "
                       "film centre: {:.3g}% of aimed rays pass, blur {:.3g} pixels (RMS); corners' relative throughput {:.4g}",
                       id, f, props.focal_length_x, props.focal_length_y,
                       props.centre.pupil_diameter, props.centre.pupil_obscuration,
                       100*props.centre.passed_fraction, blur_pixels, props.corner_relative_throughput)
        << '\n';

    return std::make_shared<telescope_t>(
        context,
        std::move(id),
        std::move(optics),
        fp,
        place,
        props,
        std::move(aim_grid),
        std::move(film).value(),
        samples_per_element,
        ray_trace || context.renderer_force_ray_tracing,
        tan_alpha,
        phase_space_extent_scale
    );
}

}


std::shared_ptr<sensor::sensor_t> wt::remote_sensing::load_telescope(std::string id,
                                                                     scene::loader::loader_t* loader,
                                                                     const scene::loader::node_t& node,
                                                                     const wt::wt_context_t &context) {
    if (stob_strict(node["polarimetric"], false))
        return load<true>(std::move(id), loader, node, context);
    return load<false>(std::move(id), loader, node, context);
}
