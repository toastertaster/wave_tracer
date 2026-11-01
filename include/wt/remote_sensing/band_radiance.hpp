/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <wt/sensor/response/response.hpp>
#include <wt/sensor/response/multichannel.hpp>
#include <wt/spectrum/spectrum.hpp>

namespace wt::remote_sensing {

/**
 * @brief Response of a multispectral sensor whose pixels record radiance in physical units: per band, the radiance
 *        weighted by the band's spectral response function,
 *        \f$ \int L_\lambda(\lambda) S(\lambda) \, d\lambda \f$
 *        (with `normalize`, divided by \f$ \int S \, d\lambda \f$: the band-averaged spectral radiance).
 *
 *        Spectral quantities in a scene (emission spectra, spectral irradiances) are densities per wavelength, in
 *        the units of `spectral_unit`, e.g. a solar spectral irradiance in W/m²/nm with `spectral_unit` nm. Films
 *        integrate samples over wavenumber (in mm⁻¹), so each channel's response is weighted by the Jacobian
 *        \f$ d\lambda/dk = 2\pi/k^2 \f$, converted to `spectral_unit`.
 *        The response functions may be given at any wavelengths; they are resampled finely, and are zero outside
 *        their range.
 */
class band_radiance_t final : public sensor::response::response_t {
public:
    enum class spectral_unit_e { nm, um };

private:
    std::vector<std::shared_ptr<spectrum::spectrum_real_t>> bands;
    sensor::response::multichannel_t weighted;   // the bands times the Jacobian
    std::vector<f_t> band_scale;                 // 1, or 1/∫S dλ with `normalize`
    std::vector<f_t> band_integral;              // ∫S dλ, in spectral_unit
    spectral_unit_e unit;
    bool normalized;

    struct resampled_t {
        std::vector<std::shared_ptr<spectrum::spectrum_real_t>> weighted;
        std::vector<f_t> integrals;
    };
    static resampled_t resample(const std::vector<std::shared_ptr<spectrum::spectrum_real_t>>& bands,
                                spectral_unit_e unit);

    band_radiance_t(std::string id,
                    std::vector<std::shared_ptr<spectrum::spectrum_real_t>> bands,
                    spectral_unit_e unit,
                    bool normalize,
                    resampled_t resampled);

public:
    band_radiance_t(std::string id,
                    std::vector<std::shared_ptr<spectrum::spectrum_real_t>> bands,
                    spectral_unit_e unit,
                    bool normalize);
    band_radiance_t(band_radiance_t&&) = default;

    [[nodiscard]] bitmap::pixel_layout_t pixel_layout() const noexcept override {
        return weighted.pixel_layout();
    }

    [[nodiscard]] const f_t f(std::uint32_t channel, const wavenumber_t& k) const noexcept override {
        return channel<band_scale.size() ? weighted.f(channel, k) * band_scale[channel] : 0;
    }

    /**
     * @brief The sum of the bands times the Jacobian, from which wavenumbers are sampled.
     */
    [[nodiscard]] const spectrum::spectrum_real_t& sensitivity() const noexcept override {
        return weighted.sensitivity();
    }

    /**
     * @brief \f$ \int S \, d\lambda \f$ of a band, in `spectral_unit`.
     */
    [[nodiscard]] f_t integral(std::uint32_t channel) const noexcept { return band_integral[channel]; }

public:
    static std::unique_ptr<sensor::response::response_t> load(std::string id,
                                                              scene::loader::loader_t* loader,
                                                              const scene::loader::node_t& node,
                                                              const wt::wt_context_t &context);

    [[nodiscard]] scene::element::info_t description() const override;
};

}
