/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#include <format>
#include <stdexcept>

#include <wt/remote_sensing/spectral_mix.hpp>

#include <wt/texture/transform.hpp>
#include <wt/scene/element/attributes.hpp>
#include <wt/scene/loader/node_readers.hpp>
#include <wt/util/logger/logger.hpp>

using namespace wt;
using namespace wt::remote_sensing;

namespace {

/** @brief The bitmap that `t` looks up, through its transforms; or nullptr. */
const texture::bitmap_t* bitmap_of(const texture::texture_t* t) noexcept {
    while (const auto* transform = dynamic_cast<const texture::transform_t*>(t))
        t = transform->get_nested_texture();
    return dynamic_cast<const texture::bitmap_t*>(t);
}

}

spectral_mix_t::spectral_mix_t(std::string id,
                               std::shared_ptr<texture::texture_t> weights,
                               std::vector<std::shared_ptr<spectrum::spectrum_real_t>> spectra)
    : texture_t(std::move(id)),
      weights_tex(std::move(weights)),
      weights_bitmap(bitmap_of(weights_tex.get())),
      spectra(std::move(spectra))
{
    if (!weights_bitmap)
        throw std::runtime_error("(spectral_mix texture) the weights must be a bitmap texture, or a transform of one");
    if (this->spectra.empty() || this->spectra.size()>max_spectra)
        throw std::runtime_error(std::format("(spectral_mix texture) one to {} spectra expected", max_spectra));
}

std::array<f_t, 4> spectral_mix_t::values(wavenumber_t k) const noexcept {
    std::array<f_t, 4> ret{};
    for (std::size_t i=0; i<spectra.size(); ++i)
        ret[i] = spectra[i]->f(k);
    return ret;
}

bool spectral_mix_t::needs_interaction_footprint() const noexcept {
    return weights_tex->needs_interaction_footprint();
}

vec2_t spectral_mix_t::resolution() const noexcept {
    return weights_tex->resolution();
}

std::shared_ptr<spectrum::spectrum_real_t> spectral_mix_t::mean_spectrum() const noexcept {
    return nullptr;
}

std::optional<f_t> spectral_mix_t::mean_value(wavenumber_t k) const noexcept {
    const auto w = weights_bitmap->get_texture()->mean_value();
    const auto s = values(k);
    return w.x*s[0] + w.y*s[1] + w.z*s[2] + w.w*s[3];
}

vec4_t spectral_mix_t::get_RGBA(const texture::texture_query_t& query) const noexcept {
    return weights_tex->get_RGBA(query);
}

vec2_t spectral_mix_t::f(const texture::texture_query_t& query) const noexcept {
    const auto w = weights_tex->get_RGBA(query);
    const auto s = values(query.k);
    return { w.x*s[0] + w.y*s[1] + w.z*s[2] + w.w*s[3], 1 };
}

scene::element::info_t spectral_mix_t::description() const {
    using namespace scene::element;
    std::vector<attribute_ptr> mixed;
    for (const auto& s : spectra)
        mixed.emplace_back(attributes::make_element(s.get()));
    return info_for_scene_element(*this, "spectral_mix", {
        { "weights", attributes::make_element(weights_tex.get()) },
        { "spectra", attributes::make_array(std::move(mixed)) },
    });
}

std::unique_ptr<texture::texture_t> spectral_mix_t::load(
        std::string id,
        scene::loader::loader_t* loader,
        const scene::loader::node_t& node,
        const wt::wt_context_t &context) {
    std::shared_ptr<texture::texture_t> weights;
    std::vector<std::shared_ptr<spectrum::spectrum_real_t>> spectra;

    for (auto& item : node.children_view()) {
    try {
        std::shared_ptr<spectrum::spectrum_real_t> spectrum;
        if (scene::loader::load_scene_element(item, "weights", weights, loader, context))
            continue;
        if (scene::loader::load_scene_element(item, spectrum, loader, context))
            spectra.emplace_back(std::move(spectrum));
        else
            logger::cwarn()
                << loader->node_description(item)
                << "(spectral_mix texture loader) unqueried node type " << item.name() << " (\"" << item["name"] << "\")" << '\n';
    } catch(const std::format_error& exp) {
        throw scene_loading_exception_t("(spectral_mix texture loader) " + std::string{ exp.what() }, item);
    }
    }

    if (!weights)
        throw scene_loading_exception_t("(spectral_mix texture loader) a 'weights' texture must be provided", node);

    std::unique_ptr<spectral_mix_t> ret;
    try {
        ret = std::make_unique<spectral_mix_t>(std::move(id), std::move(weights), std::move(spectra));
    } catch(const std::runtime_error& exp) {
        throw scene_loading_exception_t(exp.what(), node);
    }
    loader->register_resource_dependency(ret.get(), ret->weights_tex.get());
    return ret;
}
