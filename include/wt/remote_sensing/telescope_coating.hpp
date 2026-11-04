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

#include <wt/math/common.hpp>
#include <wt/spectrum/spectrum.hpp>
#include <wt/wt_context.hpp>

namespace wt::scene::loader {
class loader_t;
class node_t;
}

namespace wt::remote_sensing::optics {

/**
 * @brief Complex amplitude reflection coefficients, in wave_tracer's convention (as `wt::fresnel_reflection`'s, and
 *        `mueller_operator_t::fresnel`'s arguments): at normal incidence rp = -rs, and a bare interface with a medium
 *        of refractive index N reflects rs = (cosθ - N cosθₜ)/(cosθ + N cosθₜ).
 */
struct fresnel_coefficients_t {
    c_t rs, rp;
};

/**
 * @brief A mirror's coating, in vacuum: ideal, or a stack of thin films on a substrate.
 *
 *        An ideal coating reflects as a perfect conductor (rs = -1, rp = 1): all light, the polarization only
 *        mirrored. A multilayer coating reflects as its layers (each of a refractive index and a thickness, listed
 *        from the outermost) on its substrate (a metal's complex refractive index; or any medium's): computed by the
 *        characteristic matrices of the layers (Macleod, Thin-Film Optical Filters, ch. 2), in double precision,
 *        per wavelength and angle of incidence, in wave_tracer's convention. Without layers it is the bare
 *        substrate, by wave_tracer's own `fresnel_reflection`; with layers it tends to it as they thin.
 *
 *        Refractive indices are wave_tracer's spectra: complex, n + ik with k ≥ 0 absorbing (e.g. the data's
 *        metals, `<spectrum name="substrate" material="Al"/>`).
 */
class coating_t {
public:
    struct layer_t {
        std::shared_ptr<spectrum::spectrum_t> ior;
        /** @brief Metres. */
        double thickness;
    };

private:
    std::shared_ptr<spectrum::spectrum_t> substrate_ior;
    std::vector<layer_t> stack;

public:
    /** @brief An ideal coating. */
    coating_t() = default;
    /** @brief A multilayer coating: the layers from the outermost, on the substrate. */
    coating_t(std::shared_ptr<spectrum::spectrum_t> substrate, std::vector<layer_t> layers) noexcept
        : substrate_ior(std::move(substrate)), stack(std::move(layers))
    {}

    [[nodiscard]] inline bool is_ideal() const noexcept { return !substrate_ior; }
    [[nodiscard]] inline const auto& substrate() const noexcept { return substrate_ior; }
    [[nodiscard]] inline const auto& layers() const noexcept { return stack; }

    /**
     * @brief The reflection coefficients at wavenumber `k`, for light incident at angle θ (cosθ = `cos_theta`).
     */
    [[nodiscard]] fresnel_coefficients_t reflection(wavenumber_t k, double cos_theta) const noexcept;

    /**
     * @brief The reflectance (of unpolarized light) at wavenumber `k`, at normal incidence.
     */
    [[nodiscard]] double normal_reflectance(wavenumber_t k) const noexcept;

    [[nodiscard]] std::string description() const;

    /**
     * @brief Loads a `<coating type="ideal"/>`, or a `<coating type="multilayer">` of a `substrate` spectrum and
     *        `<layer>`s (each an `IOR` spectrum and a `thickness`), from the outermost.
     */
    static std::shared_ptr<const coating_t> load(scene::loader::loader_t* loader,
                                                 const scene::loader::node_t& node,
                                                 const wt::wt_context_t &context);
};

}
