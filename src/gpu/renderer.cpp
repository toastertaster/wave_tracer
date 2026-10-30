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
#include <functional>
#include <map>
#include <random>
#include <vector>

#include <wt/ads/bvh8w/bvh8w.hpp>
#include <wt/scene/scene.hpp>
#include <wt/integrator/plt_path/plt_path.hpp>
#include <wt/emitter/point.hpp>
#include <wt/bsdf/two_sided.hpp>
#include <wt/bsdf/composite.hpp>
#include <wt/bsdf/diffuse.hpp>
#include <wt/bsdf/scale.hpp>
#include <wt/bsdf/surface_spm.hpp>
#include <wt/interaction/surface_profile/dirac.hpp>
#include <wt/texture/constant.hpp>
#include <wt/sensor/sensor/perspective.hpp>
#include <wt/sensor/sensor/virtual_plane_sensor.hpp>
#include <wt/sensor/film/film_storage.hpp>
#include <wt/sampler/sampler.hpp>
#include <wt/scene/element/attributes.hpp>
#include <wt/util/logger/logger.hpp>
#include <wt/gpu/kernel/plt_path.hpp>

#include "to_kernel.hpp"

using namespace wt;
using namespace wt::gpu;

/*
 * Compiles a scene into the GPU kernel scene structs (wt/gpu/kernel/scene.hpp)
 * for one sensor, refusing anything the kernels do not implement; and, pass
 * by pass, draws the paths' wavenumbers and compiles what depends on them
 * (`spectral_tables_t`).
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
    out.params.backward = opts.transport_direction==bsdf::transport_e::backward ? 1 : 0;
    out.params.max_depth = opts.max_depth;
    out.params.russian_roulette = opts.RR ? 1 : 0;
    out.params.fsd = opts.FSD ? 1 : 0;
    out.params.ray_tracing_only = sensor.ray_trace_only() ? 1 : 0;
}

/**
 * @brief The emitters the sensor samples (in the order of the emitter CDF,
 *        which this compiles), which must be point emitters.
 */
std::vector<const emitter::point_t*> compile_emitters(const scene_t& scene, const sensor::sensor_t& sensor,
                                                      plt_path_scene_t& out) {
    std::vector<const emitter::point_t*> emitters;
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
        emitters.push_back(point);
        pdfs.push_back(pdf);
    }
    if (emitters.empty())
        throw gpu_error_t(std::format(
            "GPU rendering found no emitter emitting in the spectrum of sensor <{}>", sensor.get_id()));

    // inclusive CDF
    double total = 0;
    for (const auto p : pdfs) total += p;
    double cdf = 0;
    for (const auto p : pdfs) {
        cdf += p;
        out.emitter_cdf.push_back(real(cdf/total));
    }
    out.emitter_cdf.back() = 1;
    out.emitter_count = std::uint32_t(emitters.size());
    return emitters;
}

/**
 * @brief A uniform sampler with a given seed, so that GPU renders draw the
 *        same wavenumbers each time; `stratify` puts its next number in a
 *        stratum.
 */
class seeded_sampler_t final : public sampler::sampler_t {
public:
    explicit seeded_sampler_t(std::uint64_t seed) : sampler_t("gpu_spectral_sampler"), engine(seed) {}

    /** @brief The next number is uniform in [j/n, (j+1)/n). */
    void stratify(std::uint32_t j, std::uint32_t n) noexcept { stratum = j; strata = n; }

    [[nodiscard]] f_t r() noexcept override {
        const auto u = d(engine);
        if (strata==0)
            return u;
        const auto v = (f_t(stratum) + u) / f_t(strata);
        strata = 0;
        return m::min(v, std::nextafter(f_t(1), f_t(0)));
    }
    [[nodiscard]] vec2_t r2() noexcept override { return { r(), r() }; }
    [[nodiscard]] vec3_t r3() noexcept override { return { r(), r(), r() }; }
    [[nodiscard]] vec4_t r4() noexcept override { return { r(), r(), r(), r() }; }

    [[nodiscard]] scene::element::info_t description() const override {
        return scene::element::info_for_scene_element(*this, "seeded uniform");
    }

private:
    std::mt19937_64 engine;
    std::uniform_real_distribution<f_t> d;
    std::uint32_t stratum = 0, strata = 0;
};

/** @brief `IOR_has_transmission` of src/bsdf/surface_spm.cpp */
bool ior_has_transmission(const c_t ior) noexcept {
    return m::sqr(m::abs(ior.imag())) / std::norm(ior) <= f_t(1e-2);
}

kernel::material compile_bsdf(const bsdf::bsdf_t& bsdf, const wavenumber_t k) {
    kernel::material m{};
    m.type = kernel::material_absorbing;
    m.scale = 1;

    // the value of a texture, which must be constant (the GPU has no texture lookups)
    const auto constant_texture = [&](const texture::texture_t& texture, const bsdf::bsdf_t& of) {
        const auto* constant = dynamic_cast<const texture::constant_t*>(&texture);
        if (!constant)
            throw gpu_error_t(std::format(
                "GPU rendering supports only constant textures so far; BSDF <{}> has a '{}' texture",
                of.get_id(), type_of(texture)));
        return constant->get_spectrum()->f(k);
    };

    // unwrap the BSDFs that select, reorient or scale another
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
        } else if (const auto* scale = dynamic_cast<const bsdf::scale_t*>(b); scale) {
            m.scale *= real(constant_texture(*scale->scale_texture(), *scale));
            b = scale->nested_bsdf().get();
        } else {
            break;
        }
    }

    if (const auto* diffuse = dynamic_cast<const bsdf::diffuse_t*>(b); diffuse) {
        m.type = kernel::material_diffuse;
        m.reflectance = real(m::clamp01<f_t>(constant_texture(*diffuse->reflectance(), *diffuse)));
        return m;
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

/** @brief Compiles the geometry; returns the materials' BSDFs (compiled for each spectral sample). */
std::vector<const bsdf::bsdf_t*> compile_geometry(const scene_t& scene, const ads::bvh8w_t& bvh,
                                                  plt_path_scene_t& out) {
    std::vector<const bsdf::bsdf_t*> bsdfs;
    std::map<const bsdf::bsdf_t*, std::uint32_t> material_of;
    std::vector<std::uint32_t> shape_material(scene.shapes().size());
    for (std::size_t i=0; i<scene.shapes().size(); ++i) {
        const auto* bsdf = &scene.shapes()[i]->get_bsdf();
        const auto [it, inserted] = material_of.try_emplace(bsdf, std::uint32_t(bsdfs.size()));
        if (inserted)
            bsdfs.push_back(bsdf);
        shape_material[i] = it->second;
    }
    out.material_count = std::uint32_t(bsdfs.size());

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
    return bsdfs;
}

/** @brief A sensor's film, and the beams it sources at a wavenumber (`sourcing_geometry`). */
struct sensor_spectral_t {
    const sensor::film_t<2,false>* film;
    std::function<beam::sourcing_geometry_t(wavenumber_t)> sourcing;
};

template <typename Film>
void compile_film(const Film& film, plt_path_scene_t& out) {
    const auto size = film.dimensions();
    const auto components = film.pixel_layout().components;
    if (components>4)
        throw gpu_error_t("GPU rendering supports films of up to 4 components");
    if (std::uint64_t(size.x)*size.y*components > std::uint64_t(1)<<32)
        throw gpu_error_t("GPU rendering supports films of up to 2^32 values");
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
}

sensor_spectral_t compile_sensor(const sensor::sensor_t& sensor, const bool backward, plt_path_scene_t& out) {
    auto& s = out.sensor;

    if (const auto* plane = dynamic_cast<const sensor::virtual_plane_sensor_t*>(&sensor); plane) {
        // virtual_plane_sensor_t::Si, sample_direct (forward) and sample (backward)
        s.type = kernel::sensor_virtual_plane;
        const auto& f = plane->frame();
        const auto& o = plane->origin();
        const auto& extent = plane->extent();
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
        s.area = real(u::to_m2(plane->area()));
        s.recp_area = real(u::to_inv_m2(f_t(1) / plane->area()));
        s.importance = real(u::to_num(plane->importance()));
        compile_film(plane->film(), out);
        return { &plane->film(), [plane](wavenumber_t k) { return plane->sourcing_geometry(k); } };
    }

    if (const auto* perspective = dynamic_cast<const sensor::perspective_t<false>*>(&sensor); perspective) {
        // perspective_t::sample; forward transport cannot reach a point sensor
        if (!backward)
            throw gpu_error_t(std::format(
                "GPU rendering supports perspective sensors only with backward plt_path "
                "(direction=\"backward\"); sensor <{}>", sensor.get_id()));
        s.type = kernel::sensor_perspective;
        s.position = to_kernel(perspective->position());
        // glm matrices are column-major
        const auto& to_camera = perspective->film_to_camera_transform().matrix();
        for (int r=0; r<4; ++r)
        for (int c=0; c<4; ++c)
            s.to_camera[4*r+c] = real(to_camera[c][r]);
        // directions transform by the inverse transpose (transform_t::operator()(dir3_t))
        const auto dir_to_world = perspective->to_world_transform().inverse_transpose();
        for (int c=0; c<3; ++c) {
            auto e = vec3_t{ 0 };
            e[c] = 1;
            const auto column = dir_to_world(e, transform_vector);
            for (int r=0; r<3; ++r)
                s.dir_to_world[3*r+c] = real(column[r]);
        }
        s.ddir_dx = to_kernel(perspective->direction_pdvx());
        s.ddir_dy = to_kernel(perspective->direction_pdvy());
        compile_film(perspective->film(), out);
        return { &perspective->film(), [perspective](wavenumber_t k) { return perspective->sourcing_geometry(k); } };
    }

    throw gpu_error_t(std::format(
        "GPU rendering supports only virtual_plane and (non-polarimetric) perspective sensors so far; "
        "sensor <{}> is a '{}'", sensor.get_id(), type_of(sensor)));
}

/** @brief splitmix64, to seed each pass's draws. */
std::uint64_t mix_seed(std::uint64_t z) noexcept {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z>>30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z>>27)) * 0x94d049bb133111ebull;
    return z ^ (z>>31);
}

}


/**
 * @brief The paths' wavenumbers: for each pass, `samples_per_emitter`
 *        wavenumbers drawn for each emitter, as the CPU draws them
 *        (`scene_t::sample_spectrum`), with what depends on them compiled
 *        (see wt/gpu/kernel/scene.hpp). A path takes one of its emitter's
 *        draws at random, so it has the wavenumber, and weight, of a CPU path.
 *
 *        A pass's paths share its draws, so their spectral error is common:
 *        independent draws would tint a pass by about 1/sqrt(draws). The
 *        draws are stratified instead (draw j's random number is in the j-th
 *        of `samples_per_emitter` strata; the spectra draw by inverting their
 *        CDFs), which cover the spectrum evenly. A sensor sensitive to a
 *        single wavelength needs just one draw per emitter, for all passes.
 */
class wt::gpu::spectral_tables_t {
public:
    /** @brief Draws per pass, over all emitters, for sensors sensitive to more than one wavelength. */
    static constexpr std::uint32_t samples_per_table = 4096;

    spectral_tables_t(const scene_t& scene, const sensor::sensor_t& sensor, const bool backward,
                      sensor_spectral_t sensor_spectral,
                      std::vector<const emitter::point_t*> emitters,
                      std::vector<const bsdf::bsdf_t*> bsdfs)
        : scene(scene), sensor(sensor), backward(backward),
          sensor_spectral(std::move(sensor_spectral)),
          emitters(std::move(emitters)), bsdfs(std::move(bsdfs))
    {
        const bool single_wavelength = sensor.sensitivity_spectrum().wavenumber_range().length()==zero;
        per_emitter = single_wavelength ? 1 :
            std::max<std::uint32_t>(1, samples_per_table / std::uint32_t(this->emitters.size()));
        fixed = single_wavelength;
        build(0);
    }

    [[nodiscard]] std::uint32_t samples_per_emitter() const noexcept { return per_emitter; }

    /** @brief The table for the pass starting with sample `pass` (the same for all, for a single wavelength). */
    [[nodiscard]] const spectral_table_t& table(std::uint64_t pass) {
        if (!fixed && pass!=built_for)
            build(pass);
        return current;
    }

private:
    void build(std::uint64_t pass) {
        current.samples.clear();
        current.emitters.clear();
        current.materials.clear();
        seeded_sampler_t sampler{ mix_seed(pass) };
        for (const auto* emitter : emitters)
        for (std::uint32_t j=0; j<per_emitter; ++j) {
            sampler.stratify(j, per_emitter);
            compile(scene.sample_spectrum(&sensor, emitter, sampler));
        }
        built_for = pass;
    }

    /** @brief Appends the spectral sample of wavenumber sample `ws`. */
    void compile(const wavenumber_sample_t& ws) {
        const auto k = ws.k;

        // spectral (importance) sampling weight: plt_path::integrate_forward, integrate_backward
        const wavenumber_t recp_spectral_pd = backward && ws.wpd.is_discrete() ?
            f_t(1) / wavenumber_density_t{ ws.wpd.mass() * u::mm } :
            f_t(1) / scene.sum_spectral_pdf_for_all_emitters(&sensor, k);
        // a wavenumber no emitter emits at contributes nothing
        const auto recp = recp_spectral_pd>zero && m::isfinite(recp_spectral_pd) ?
            recp_spectral_pd : wavenumber_t::zero();

        kernel::spectral_sample s{};
        s.k = real(u::to_inv_m(k));
        if (backward) {
            // the beams the sensor sources, which must be isotropic cones
            const auto sourcing = sensor_spectral.sourcing(k);
            if (sourcing.surface || sourcing.initial_spatial_lengths.x!=sourcing.initial_spatial_lengths.y)
                throw gpu_error_t(std::format(
                    "GPU rendering supports only sensors that source isotropic beams; sensor <{}> does not",
                    sensor.get_id()));
            s.sensor_tan_alpha = real(sourcing.tan_alpha);
            s.sensor_x0 = real(u::to_m(sourcing.initial_spatial_lengths.x));
        }
        const auto& film = *sensor_spectral.film;
        const auto components = film.pixel_layout().components;
        for (std::uint32_t c=0; c<4; ++c)
            s.response.c[c] = c<components ? real(film.response()->f(c, k)) : real(0);
        current.samples.push_back(s);

        for (const auto* point : emitters) {
            const auto sourcing = point->sourcing_geometry(k);
            const auto intensity = point->spectral_radiant_intensity(k) * recp;
            current.emitters.push_back(kernel::point_emitter{
                .position = to_kernel(point->get_position()),
                .tan_alpha = real(sourcing.tan_alpha),
                .x0 = real(u::to_m(sourcing.initial_spatial_lengths.x)),
                // point_t::sample
                .flux = real(u::to_W(intensity * (m::four_pi * u::ang::sr))),
                // point_t::sample_direct
                .intensity = real(u::to_W(intensity * u::ang::sr)),
                .pmf = real(scene.pdf_emitter(&sensor, point)),
            });
        }

        for (const auto* bsdf : bsdfs)
            current.materials.push_back(compile_bsdf(*bsdf, k));
    }

    const scene_t& scene;
    const sensor::sensor_t& sensor;
    bool backward;
    sensor_spectral_t sensor_spectral;
    std::vector<const emitter::point_t*> emitters;
    std::vector<const bsdf::bsdf_t*> bsdfs;

    std::uint32_t per_emitter;
    bool fixed;
    spectral_table_t current;
    std::uint64_t built_for = 0;
};


renderer_t::renderer_t(const device_t& device,
                       const scene_t& scene,
                       const sensor::sensor_t& sensor,
                       const ads::ads_t& ads) {
    const auto* bvh = dynamic_cast<const ads::bvh8w_t*>(&ads);
    if (!bvh)
        throw gpu_error_t("the GPU supports only the bvh8w acceleration structure");

    plt_path_scene_t compiled;
    compile_integrator(scene, sensor, compiled);
    backward = compiled.params.backward!=0;
    auto sensor_spectral = compile_sensor(sensor, backward, compiled);
    auto emitters = compile_emitters(scene, sensor, compiled);
    auto bsdfs = compile_geometry(scene, *bvh, compiled);
    // the first pass's table: compiles the BSDFs, refusing those the GPU does not support
    spectra = std::make_unique<spectral_tables_t>(scene, sensor, backward, std::move(sensor_spectral),
                                                  std::move(emitters), std::move(bsdfs));
    compiled.samples_per_emitter = spectra->samples_per_emitter();

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
    // every sample of every element is one path: path `s*paths_per_spe + e` is sample `s` of element `e`
    // (forward paths ignore their element, as the CPU's do)
    const auto& table = spectra->table(first);
    std::uint32_t status;
    try {
        status = backend->trace(first*paths_per_spe, count*paths_per_spe, /* seed */ 0, table);
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

    // as the CPU's integrators write: backward, block splats; forward, direct splats
    std::vector<double> values(film_values);
    if (backward) {
        std::vector<double> weights(film_values);
        backend->read_film(values.data(), weights.data());
        storage->set_image(values.data(), weights.data());
    } else {
        backend->read_film(values.data(), nullptr);
        storage->set_light_image(values.data());
    }
}
