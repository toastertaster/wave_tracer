/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#include <format>

#include <wt/remote_sensing/band_radiance.hpp>

#include <wt/scene/element/attributes.hpp>
#include <wt/scene/loader/node_readers.hpp>
#include <wt/spectrum/piecewise_linear.hpp>
#include <wt/math/common.hpp>
#include <wt/util/logger/logger.hpp>

using namespace wt;
using namespace wt::remote_sensing;

namespace {

// resampling of the response functions: at most this wavelength step, and at least this many steps per band
constexpr double max_step_nm = .5;
constexpr std::size_t min_steps = 256;

/** @brief Wavelengths per spectral_unit, in mm. */
double unit_in_mm(band_radiance_t::spectral_unit_e unit) noexcept {
    return unit==band_radiance_t::spectral_unit_e::nm ? 1e-6 : 1e-3;
}

struct resampled_band_t {
    std::shared_ptr<spectrum::spectrum_real_t> weighted;
    double integral;    // ∫S dλ, in spectral_unit
};

/**
 * @brief Samples a band's response finely in wavelength, times the Jacobian dλ/dk (λ in spectral_unit, k in mm⁻¹).
 *        Outside its range the response falls linearly to zero within one step, so that a sum of disjoint bands is
 *        zero between them. Not more abruptly: wavenumbers are sampled from binned products of spectra (see
 *        scene_build_sensor_sampling_data.cpp), whose bins would not resolve a sharper edge, making samples just
 *        inside it very unlikely for their weight.
 */
resampled_band_t resample_band(const spectrum::spectrum_real_t& band, band_radiance_t::spectral_unit_e unit) {
    const auto krange = band.wavenumber_range();
    const double kmin = u::to_inv_mm(krange.min), kmax = u::to_inv_mm(krange.max);
    if (!(kmin>0) || !(kmax>kmin) || !std::isfinite(kmax))
        throw std::runtime_error("(band_radiance response) a band's response function must span a finite range of wavelengths");

    const double lmin = m::two_pi/kmax, lmax = m::two_pi/kmin;     // mm
    const auto steps = std::max(min_steps, std::size_t(std::ceil((lmax-lmin) / (max_step_nm*1e-6))));
    const double mm_per_unit = unit_in_mm(unit);

    std::vector<vec2_t> values;
    values.reserve(steps+3);
    double integral = 0, prev_s = 0;
    for (std::size_t i=0;i<=steps;++i) {
        const double l = lmin + (lmax-lmin) * double(i)/steps;     // mm
        const double k = m::two_pi/l;
        // stay inside the band's range despite rounding
        const double s = band.f(m::clamp(k, kmin, kmax) / u::mm);
        const double jacobian = m::two_pi/(k*k) / mm_per_unit;   // dλ/dk, spectral_unit per mm⁻¹
        values.emplace_back(f_t(k), f_t(s*jacobian));
        if (i>0)
            integral += (s+prev_s)/2 * (lmax-lmin)/steps / mm_per_unit;
        prev_s = s;
    }
    // the ramps to zero, one step outside the range, and their part of ∫S dλ
    const double step = (lmax-lmin)/steps;
    values.emplace_back(f_t(m::two_pi/(lmin-step)), 0);
    values.emplace_back(f_t(m::two_pi/(lmax+step)), 0);
    integral += (band.f(kmax/u::mm) + band.f(kmin/u::mm))/2 * step / mm_per_unit;
    std::ranges::sort(values, [](const auto& a, const auto& b) { return a.x<b.x; });

    if (!(integral>0))
        throw std::runtime_error("(band_radiance response) a band's response function integrates to zero");

    return {
        std::make_shared<spectrum::piecewise_linear_t>(band.get_id()+"_jacobian",
                                                       piecewise_linear_distribution_t{ std::move(values) }),
        integral,
    };
}

}


band_radiance_t::resampled_t band_radiance_t::resample(
        const std::vector<std::shared_ptr<spectrum::spectrum_real_t>>& bands,
        spectral_unit_e unit) {
    if (bands.empty())
        throw std::runtime_error("(band_radiance response) at least one band must be provided");
    resampled_t ret;
    for (const auto& b : bands) {
        auto r = resample_band(*b, unit);
        ret.weighted.emplace_back(std::move(r.weighted));
        ret.integrals.emplace_back(f_t(r.integral));
    }
    return ret;
}

band_radiance_t::band_radiance_t(std::string id,
                                 std::vector<std::shared_ptr<spectrum::spectrum_real_t>> bands,
                                 spectral_unit_e unit,
                                 bool normalize)
    // bands are shared pointers: copying them is cheap, and keeps them valid for resample()
    : band_radiance_t(id, bands, unit, normalize, resample(bands, unit))
{}

band_radiance_t::band_radiance_t(std::string id,
                                 std::vector<std::shared_ptr<spectrum::spectrum_real_t>> bands,
                                 spectral_unit_e unit,
                                 bool normalize,
                                 resampled_t resampled)
    : response_t(id, nullptr),
      bands(std::move(bands)),
      weighted(id+"_jacobian", std::move(resampled.weighted)),
      band_integral(std::move(resampled.integrals)),
      unit(unit),
      normalized(normalize)
{
    for (const auto integral : band_integral)
        band_scale.emplace_back(normalize ? 1/integral : 1);
}

scene::element::info_t band_radiance_t::description() const {
    using namespace scene::element;

    std::vector<attribute_ptr> channels, integrals;
    for (std::size_t c=0;c<bands.size();++c) {
        channels.emplace_back(attributes::make_element(bands[c].get()));
        integrals.emplace_back(attributes::make_scalar(band_integral[c]));
    }
    const std::string unit_name = unit==spectral_unit_e::nm ? "nm" : "µm";
    return info_for_scene_element(*this, "band_radiance", {
        { "bands", attributes::make_array(std::move(channels)) },
        { "band integrals", attributes::make_array(std::move(integrals)) },
        { "spectral unit", attributes::make_string(unit_name) },
        { "records", attributes::make_string(normalized ?
                "band-averaged spectral radiance (per " + unit_name + ")" :
                "band radiance (radiance per " + unit_name + " times " + unit_name + ")") },
    });
}

std::unique_ptr<sensor::response::response_t> band_radiance_t::load(
        std::string id,
        scene::loader::loader_t* loader,
        const scene::loader::node_t& node,
        const wt::wt_context_t &context) {
    std::vector<std::shared_ptr<spectrum::spectrum_real_t>> bands;
    std::string unit = "nm";
    bool normalize = false;

    for (auto& item : node.children_view()) {
    try {
        std::shared_ptr<spectrum::spectrum_real_t> band;
        if (scene::loader::load_scene_element(item, band, loader, context))
            bands.emplace_back(std::move(band));
        else if (!scene::loader::read_attribute(item, "spectral_unit", unit) &&
                 !scene::loader::read_attribute(item, "normalize", normalize))
            logger::cwarn()
                << loader->node_description(item)
                << "(band_radiance response loader) unqueried node type " << item.name() << " (\"" << item["name"] << "\")" << '\n';
    } catch(const std::format_error& exp) {
        throw scene_loading_exception_t("(band_radiance response loader) " + std::string{ exp.what() }, item);
    }
    }

    if (bands.empty())
        throw scene_loading_exception_t("(band_radiance response loader) bands must be provided as child spectra", node);
    if (unit!="nm" && unit!="um")
        throw scene_loading_exception_t("(band_radiance response loader) 'spectral_unit' must be 'nm' or 'um'", node);

    try {
        return std::make_unique<band_radiance_t>(
            std::move(id), std::move(bands),
            unit=="nm" ? spectral_unit_e::nm : spectral_unit_e::um,
            normalize);
    } catch(const std::runtime_error& exp) {
        throw scene_loading_exception_t(exp.what(), node);
    }
}
