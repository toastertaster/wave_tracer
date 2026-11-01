/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#include <cassert>

#include <wt/scene/scene.hpp>
#include <wt/emitter/emitter.hpp>

using namespace wt;
using namespace wt::scene;


const emitter::emitter_t* scene_sensor_t::emitter_sampling_data_t::sample(const scene_t& parent, sampler::sampler_t& sampler) const noexcept {
    const auto i = emitters_power_distribution.icdf(sampler.r());
    const auto* ret = parent.emitters()[i].get();
    assert(ret->scene_emitter_idx == i);

    return ret;
}


scene_sensor_t::scene_sensor_t(const wt_context_t& ctx,
                               std::shared_ptr<sensor::sensor_t> sensor,
                               const scene_t* scene)
    : sensor(std::move(sensor)),
      emitter_sampler(scene_sensor_t::emitter_sampling_data_t::build_sampling_data(ctx, this->sensor.get(), scene))
{}

emitter_wavenumber_sample_t scene_sensor_t::sample_emitter_and_spectrum(
        const scene_t& parent, 
        sampler::sampler_t& sampler) const noexcept {
    // choose an emitter
    const auto* emitter = emitter_sampler.sample(parent, sampler);
    const auto ep = pdf_emitter(parent, emitter);
    assert(ep>zero);

    return {
        .emitter = emitter,
        .emitter_pdf = ep,
        .wavenumber = sample_spectrum(emitter, sampler),
    };
}

namespace {

/** @brief A sampler whose next number is given, and then those of another sampler. */
class given_first_sampler_t final : public sampler::sampler_t {
public:
    given_first_sampler_t(sampler::sampler_t& rest, f_t first) noexcept
        : sampler_t("given_first"), rest(rest), first(first) {}

    [[nodiscard]] f_t r() noexcept override {
        if (!used) { used = true; return first; }
        return rest.r();
    }
    [[nodiscard]] vec2_t r2() noexcept override { return { r(), r() }; }
    [[nodiscard]] vec3_t r3() noexcept override { return { r(), r(), r() }; }
    [[nodiscard]] vec4_t r4() noexcept override { return { r(), r(), r(), r() }; }

    [[nodiscard]] scene::element::info_t description() const override {
        return scene::element::info_for_scene_element(*this, "given first");
    }

private:
    sampler::sampler_t& rest;
    f_t first;
    bool used = false;
};

}

emitter_wavenumber_sample_t scene_sensor_t::sample_emitter_and_spectrum(
        const scene_t& parent, 
        sampler::sampler_t& sampler,
        f_t spectral_u) const noexcept {
    const auto* emitter = emitter_sampler.sample(parent, sampler);
    const auto ep = pdf_emitter(parent, emitter);
    assert(ep>zero);

    auto spectral_sampler = given_first_sampler_t{ sampler, spectral_u };
    return {
        .emitter = emitter,
        .emitter_pdf = ep,
        .wavenumber = sample_spectrum(emitter, spectral_sampler),
    };
}

wavenumber_sample_t scene_sensor_t::sample_spectrum(
        const emitter::emitter_t* emitter,
        sampler::sampler_t& sampler) const noexcept {
    const auto wavenumber_sample = emitter_sampler.sample_wavenumber(sampler, emitter);
    const auto k = wavenumber_sample.x / u::mm;

    const auto wpd = wavenumber_sample.measure==measure_e::discrete ? 
        wavenumber_sampling_pd_t::discrete(wavenumber_sample.pdf) :
        wavenumber_sampling_pd_t{ wavenumber_sample.pdf * u::mm };

    return {
        .k = k,
        .wpd = wpd,
    };
}
