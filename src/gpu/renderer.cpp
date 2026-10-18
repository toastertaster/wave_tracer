/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#include <wt/gpu/renderer.hpp>
#include <wt/gpu/backend.hpp>

#include <format>
#include <map>
#include <vector>

#include <wt/ads/bvh8w/bvh8w.hpp>
#include <wt/scene/scene.hpp>
#include <wt/integrator/plt_path/plt_path.hpp>
#include <wt/emitter/point.hpp>
#include <wt/bsdf/two_sided.hpp>
#include <wt/bsdf/composite.hpp>
#include <wt/bsdf/surface_spm.hpp>
#include <wt/interaction/surface_profile/dirac.hpp>
#include <wt/sensor/sensor/virtual_plane_sensor.hpp>
#include <wt/sensor/film/film_storage.hpp>
#include <wt/util/logger/logger.hpp>
#include <wt/gpu/kernel/plt_path.hpp>

#include "to_kernel.hpp"

using namespace wt;
using namespace wt::gpu;

/*
 * Compiles a scene into the GPU kernel scene structs (wt/gpu/kernel/scene.hpp)
 * for one sensor and wavenumber, refusing anything the kernels do not implement.
 */

namespace {

using kernel::real;

std::string type_of(const scene::scene_element_t& element) {
    return element.description().type;
}

void compile_integrator(const scene_t& scene, const sensor::sensor_t& sensor,
                        plt_path_scene_t& out) {
    const auto& integrator = scene.integrator();
    const auto* plt_path = dynamic_cast<const integrator::plt_path_t*>(&integrator);
    if (!plt_path)
        throw gpu_error_t(std::format(
            "GPU rendering supports only the 'plt_path' integrator, not '{}'", type_of(integrator)));

    const auto& opts = plt_path->get_options();
    if (opts.transport_direction!=bsdf::transport_e::forward)
        throw gpu_error_t("GPU rendering supports only forward plt_path (direction=\"forward\") so far");

    out.params.max_depth = opts.max_depth;
    out.params.russian_roulette = opts.RR ? 1 : 0;
    out.params.fsd = opts.FSD ? 1 : 0;
    out.params.ray_tracing_only = sensor.ray_trace_only() ? 1 : 0;
}

/**
 * @brief The wavenumber all paths have: the sensor must be sensitive to a single
 *        wavelength, so that every emitter sample has it (as the CPU samples it).
 */
wavenumber_t rendered_wavenumber(const scene_t& scene, const sensor::sensor_t& sensor) {
    const auto krange = sensor.sensitivity_spectrum().wavenumber_range();
    if (krange.length()!=zero)
        throw gpu_error_t(std::format(
            "GPU rendering needs a sensor sensitive to a single wavelength (as coverage sensors are), "
            "but sensor <{}> has a sensitivity spectrum spanning a range of wavelengths", sensor.get_id()));

    // the CPU's wavenumber sample, exactly (the spectral sampling densities are evaluated at it)
    const auto sample = scene.sample_emitter_and_spectrum(&sensor);
    const auto k = sample.wavenumber.k;
    if (!sample.wavenumber.wpd.is_discrete() || !(m::abs(k-krange.min) <= krange.min*f_t(1e-5)))
        throw gpu_error_t("GPU rendering expected the scene to sample only the sensor's wavelength");
    return k;
}

void compile_emitters(const scene_t& scene, const sensor::sensor_t& sensor, const wavenumber_t k,
                      plt_path_scene_t& out) {
    // spectral sampling weight (plt_path::integrate_forward)
    const auto spectral_pd = scene.sum_spectral_pdf_for_all_emitters(&sensor, k);
    if (!(spectral_pd>zero) || !m::isfinite(spectral_pd))
        throw gpu_error_t("GPU rendering found no emitter emitting at the sensor's wavelength");
    const auto recp_spectral_pd = f_t(1) / spectral_pd;

    std::vector<double> pdfs;
    for (const auto& e : scene.emitters()) {
        // emitters that are never sampled for this sensor do not matter
        const auto pdf = scene.pdf_emitter(&sensor, e.get());
        if (!(pdf>0))
            continue;

        const auto* point = dynamic_cast<const emitter::point_t*>(e.get());
        if (!point)
            throw gpu_error_t(std::format(
                "GPU rendering supports only point emitters so far; emitter <{}> is a '{}'",
                e->get_id(), type_of(*e)));

        // point_t::sample
        const auto sourcing = point->sourcing_geometry(k);
        const auto flux = point->spectral_radiant_intensity(k) * (m::four_pi * u::ang::sr) * recp_spectral_pd;
        out.emitters.push_back(kernel::point_emitter{
            .position = to_kernel(point->get_position()),
            .tan_alpha = real(sourcing.tan_alpha),
            .x0 = real(u::to_m(sourcing.initial_spatial_lengths.x)),
            .flux = real(u::to_W(flux)),
        });
        pdfs.push_back(pdf);
    }
    if (out.emitters.empty())
        throw gpu_error_t("GPU rendering found no emitter emitting at the sensor's wavelength");

    // inclusive CDF
    double total = 0;
    for (const auto p : pdfs) total += p;
    double cdf = 0;
    for (const auto p : pdfs) {
        cdf += p;
        out.emitter_cdf.push_back(real(cdf/total));
    }
    out.emitter_cdf.back() = 1;
}

/** @brief `IOR_has_transmission` of src/bsdf/surface_spm.cpp */
bool ior_has_transmission(const c_t ior) noexcept {
    return m::sqr(m::abs(ior.imag())) / std::norm(ior) <= f_t(1e-2);
}

kernel::material compile_bsdf(const bsdf::bsdf_t& bsdf, const wavenumber_t k) {
    kernel::material m{};
    m.type = kernel::material_absorbing;

    // unwrap the BSDFs that select or reorient another
    const bsdf::bsdf_t* b = &bsdf;
    for (;;) {
        if (const auto* two_sided = dynamic_cast<const bsdf::two_sided_t*>(b); two_sided) {
            m.two_sided = 1;
            b = two_sided->nested_bsdf().get();
        } else if (const auto* composite = dynamic_cast<const bsdf::composite_t*>(b); composite) {
            b = composite->bsdf_for(k);
            // no BSDF for this wavelength: paths end here
            if (!b)
                return m;
        } else {
            break;
        }
    }

    if (const auto* spm = dynamic_cast<const bsdf::surface_spm_t*>(b); spm) {
        if (!dynamic_cast<const surface_profile::dirac_t*>(spm->surface_profile().get()))
            throw gpu_error_t(std::format(
                "GPU rendering supports only smooth (Dirac surface profile) surface_spm BSDFs so far; "
                "BSDF <{}> has a '{}' surface profile", spm->get_id(), type_of(*spm->surface_profile())));

        const auto eta = spm->IOR(k);
        m.type = kernel::material_specular;
        m.has_transmission = ior_has_transmission(eta) ? 1 : 0;
        m.eta = { real(eta.real()), real(eta.imag()) };
        m.reflection_scale = real(spm->reflectivity_scale(k));
        m.transmission_scale = real(spm->transmissivity_scale(k));
        return m;
    }

    throw gpu_error_t(std::format(
        "GPU rendering does not support '{}' BSDFs yet (BSDF <{}>)", type_of(*b), b->get_id()));
}

void compile_geometry(const scene_t& scene, const ads::bvh8w_t& bvh, const wavenumber_t k,
                      plt_path_scene_t& out) {
    std::map<const bsdf::bsdf_t*, std::uint32_t> material_of;
    std::vector<std::uint32_t> shape_material(scene.shapes().size());
    for (std::size_t i=0; i<scene.shapes().size(); ++i) {
        const auto* bsdf = &scene.shapes()[i]->get_bsdf();
        const auto [it, inserted] = material_of.try_emplace(bsdf, std::uint32_t(out.materials.size()));
        if (inserted)
            out.materials.push_back(compile_bsdf(*bsdf, k));
        shape_material[i] = it->second;
    }

    // shading data in the BVH's triangle order (see flatten() in src/gpu/gpu.cpp)
    const auto edge_id = [](const ads::tuid_t e) { return e ? std::uint32_t(e.uid) : kernel::no_edge; };
    out.tris.resize(bvh.triangles_count());
    for (std::size_t i=0; i<out.tris.size(); ++i) {
        const auto& t = bvh.tri(ads::tuid_t{ ads::idx_t(i) });
        const auto& mesh_tri = scene.shapes()[t.shape_idx]->get_mesh().triangle(t.shape_tri_idx);

        auto& s = out.tris[i];
        for (int v=0; v<3; ++v)
            s.n[v] = to_kernel(dir3_t{ mesh_tri.n[v] });
        s.dpdu = to_kernel(mesh_tri.tangent_frame.dpdu);
        s.material = shape_material[t.shape_idx];
        s.edges[0] = edge_id(t.edge_ab);
        s.edges[1] = edge_id(t.edge_bc);
        s.edges[2] = edge_id(t.edge_ca);
    }

    // edges
    if (bvh.edges_count()>=std::size_t(1)<<31)
        throw gpu_error_t("GPU rendering supports up to 2^31 edges");
    const auto* first_tri = &bvh.tri(ads::tuid_t{ 0 });
    const auto tri_index = [&](const ads::tri_t* t) {
        return t ? std::uint32_t(t - first_tri) : kernel::no_triangle;
    };
    out.edges.resize(bvh.edges_count());
    for (std::size_t i=0; i<out.edges.size(); ++i) {
        const auto& e = bvh.edge(std::uint32_t(i));
        out.edges[i] = kernel::edge{
            .a = to_kernel(e.a), .b = to_kernel(e.b),
            .n1 = to_kernel(e.n1), .t1 = to_kernel(e.t1),
            .n2 = to_kernel(e.n2), .t2 = to_kernel(e.t2),
            .alpha = real(u::to_rad(e.alpha)),
            .tri1 = tri_index(e.tri1),
            .tri2 = tri_index(e.tri2),
        };
    }
}

void compile_sensor(const sensor::sensor_t& sensor, const wavenumber_t k, plt_path_scene_t& out) {
    const auto* plane = dynamic_cast<const sensor::virtual_plane_sensor_t*>(&sensor);
    if (!plane)
        throw gpu_error_t(std::format(
            "GPU rendering supports only virtual_plane sensors so far; sensor <{}> is a '{}'",
            sensor.get_id(), type_of(sensor)));

    // virtual_plane_sensor_t::Si
    const auto& f = plane->frame();
    const auto& o = plane->origin();
    const auto& extent = plane->extent();
    auto& s = out.sensor;
    s.origin = to_kernel(o);
    s.f = to_kernel(f);
    s.quad[0] = to_kernel(o);
    s.quad[1] = to_kernel(o + extent.x*f.t);
    s.quad[2] = to_kernel(o + extent.y*f.b);
    s.quad[3] = to_kernel(o + extent.x*f.t + extent.y*f.b);
    const auto element_extent = plane->element_extent();
    const auto recp_element_extent = f_t(1) / element_extent;
    s.extent = { real(u::to_m(extent.x)), real(u::to_m(extent.y)) };
    s.element_extent = { real(u::to_m(element_extent.x)), real(u::to_m(element_extent.y)) };
    s.recp_element_extent = { real(u::to_inv_m(recp_element_extent.x)), real(u::to_inv_m(recp_element_extent.y)) };
    s.recp_area = real(u::to_inv_m2(f_t(1) / plane->area()));
    s.importance = real(u::to_num(plane->importance()));

    // film
    const auto& film = plane->film();
    const auto size = film.dimensions();
    const auto components = film.pixel_layout().components;
    if (components>4)
        throw gpu_error_t("GPU rendering supports films of up to 4 components");
    if (film.reconstruction_filter_radius()>kernel::max_rf_radius)
        throw gpu_error_t(std::format(
            "GPU rendering supports film reconstruction filters up to {} elements wide; "
            "reduce the film's rfilter_scale", kernel::max_rf_radius));

    auto& fd = out.film;
    fd.width = size.x;
    fd.height = size.y;
    fd.components = components;
    fd.rf_radius = kernel::i32(film.reconstruction_filter_radius());
    fd.rf_sigma = real(film.reconstruction_filter_stddev());
    for (std::uint32_t c=0; c<4; ++c)
        fd.response[c] = c<components ? real(film.response()->f(c, k)) : real(0);
}

}


renderer_t::renderer_t(const device_t& device,
                       const scene_t& scene,
                       const sensor::sensor_t& sensor,
                       const ads::ads_t& ads) {
    const auto* bvh = dynamic_cast<const ads::bvh8w_t*>(&ads);
    if (!bvh)
        throw gpu_error_t("the GPU supports only the bvh8w acceleration structure");

    plt_path_scene_t compiled;
    compile_integrator(scene, sensor, compiled);
    const auto k = rendered_wavenumber(scene, sensor);
    compiled.params.k = real(u::to_inv_m(k));
    compile_sensor(sensor, k, compiled);
    compile_emitters(scene, sensor, k, compiled);
    compile_geometry(scene, *bvh, k, compiled);

    paths_per_spe = std::uint64_t(compiled.film.width) * compiled.film.height;
    film_values = std::size_t(paths_per_spe) * compiled.film.components;

    geometry = device.upload(ads);
    try {
        backend = geometry->backend->create_plt_path_renderer(compiled);
    } catch (const gpu_error_t&) {
        throw;
    } catch (const std::exception& e) {
        throw gpu_error_t(std::string("cannot upload the scene to the GPU: ") + e.what());
    }
}

renderer_t::~renderer_t() noexcept = default;

void renderer_t::render(std::uint64_t first, std::uint64_t count) {
    // every sample of every element is one path (the CPU ignores which element a forward path belongs to)
    std::uint32_t status;
    try {
        status = backend->trace(first*paths_per_spe, count*paths_per_spe, /* seed */ 0);
    } catch (const std::exception& e) {
        throw gpu_error_t(std::string("GPU rendering failed: ") + e.what());
    }
    check_status(status);
}

void renderer_t::finish() {
    std::uint32_t status;
    try {
        status = backend->finish();
    } catch (const std::exception& e) {
        throw gpu_error_t(std::string("GPU rendering failed: ") + e.what());
    }
    check_status(status);
}

void renderer_t::check_status(std::uint32_t status) {
    if (status & kernel::status_stack_overflow)
        throw gpu_error_t("GPU BVH traversal overflowed its stack");
    if ((status & kernel::status_iteration_limit) && !warned_iteration_limit) {
        warned_iteration_limit = true;
        logger::cwarn()
            << "(gpu) some paths took more than " << kernel::max_path_iterations
            << " steps and were stopped" << '\n';
    }
}

void renderer_t::write_film(sensor::film_storage_handle_t& film) const {
    auto* storage = dynamic_cast<sensor::film_storage_t<false, 2>*>(&film);
    if (!storage || storage->pixel_layout().components*paths_per_spe!=film_values)
        throw gpu_error_t("GPU rendering expected the sensor's 2D scalar film");

    std::vector<double> values(film_values);
    backend->read_film(values.data());
    storage->set_light_image(values.data());
}
