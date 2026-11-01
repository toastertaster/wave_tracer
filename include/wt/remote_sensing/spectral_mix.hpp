/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <wt/texture/texture.hpp>
#include <wt/texture/bitmap.hpp>
#include <wt/spectrum/spectrum.hpp>

namespace wt::remote_sensing {

/**
 * @brief A linear mixture of spectra (e.g. measured reflectances of materials) whose weights vary over a surface:
 *        \f$ \sum_i w_i(uv) S_i(\lambda) \f$, with the weights \f$ w_i \f$ the R, G, B and A channels of a bitmap
 *        (as `texture_t::get_RGBA` gives them: a grey bitmap's value in R, G and B, and A 1 without alpha).
 *        Up to four spectra.
 *
 *        Remote sensing's linear mixture model: the weights are the materials' abundances (fractions of the
 *        surface); they need not sum to one, and so also carry the materials' variations in brightness.
 */
class spectral_mix_t final : public texture::texture_t {
public:
    static constexpr std::size_t max_spectra = 4;

private:
    std::shared_ptr<texture::texture_t> weights_tex;
    const texture::bitmap_t* weights_bitmap;      // the bitmap that weights_tex looks up (transformed)
    std::vector<std::shared_ptr<spectrum::spectrum_real_t>> spectra;

public:
    /**
     * @param weights a bitmap texture, or a transform of one
     * @param spectra one to four
     */
    spectral_mix_t(std::string id,
                   std::shared_ptr<texture::texture_t> weights,
                   std::vector<std::shared_ptr<spectrum::spectrum_real_t>> spectra);
    spectral_mix_t(spectral_mix_t&&) = default;

    /** @brief The weights' texture. */
    [[nodiscard]] const texture::texture_t* weights() const noexcept { return weights_tex.get(); }

    /** @brief The spectra's values at `k`, per RGBA channel of the weights (0 beyond the spectra). */
    [[nodiscard]] std::array<f_t, 4> values(wavenumber_t k) const noexcept;

    [[nodiscard]] bool needs_interaction_footprint() const noexcept override;
    [[nodiscard]] vec2_t resolution() const noexcept override;
    /** @brief Not computed: nullptr. */
    [[nodiscard]] std::shared_ptr<spectrum::spectrum_real_t> mean_spectrum() const noexcept override;
    /** @brief The mixture with the weights' bitmap's mean weights. */
    [[nodiscard]] std::optional<f_t> mean_value(wavenumber_t k) const noexcept override;
    /** @brief The weights. */
    [[nodiscard]] vec4_t get_RGBA(const texture::texture_query_t& query) const noexcept override;
    /** @brief The mixture at `query.k`, and alpha 1. */
    [[nodiscard]] vec2_t f(const texture::texture_query_t& query) const noexcept override;

public:
    static std::unique_ptr<texture::texture_t> load(std::string id,
                                                    scene::loader::loader_t* loader,
                                                    const scene::loader::node_t& node,
                                                    const wt::wt_context_t &context);

    [[nodiscard]] scene::element::info_t description() const override;
};

}
