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
#include <cmath>
#include <vector>

#include <wt/math/common.hpp>

namespace wt::remote_sensing::optics {

/**
 * @brief A mirror's scatter from its surface's roughness, by the ABg (Harvey) model of stray-light analysis (as in
 *        FRED, ASAP or Zemax): the BSDF, relative to the coating's specular reflectance, is
 *            BSDF = A / (B + |β - β₀|^g),
 *        β and β₀ the projections, on the surface's tangent plane, of the scattered direction and of the specular
 *        one (so that BSDF cosθₛ dωₛ = BSDF d²β). The total integrated scatter at incidence |β₀| = sinθᵢ,
 *        TIS(β₀) = ∫ BSDF d²β over the unit disc, is the fraction of the reflected light that is scattered
 *        rather than specular.
 *
 *        Scattered directions are sampled in β: |β - β₀| from a table of the radial distribution 2πr BSDF(r)
 *        over r in [0,2] (log-spaced bins, uniform by area within each), its angle uniformly.
 */
class abg_scatter_t {
public:
    static constexpr std::size_t tis_samples = 65;       // TIS at |β₀| = 0, 1/64, ..., 1
    static constexpr std::size_t sampling_bins = 256;
    static constexpr double max_offset = 2;               // |β - β₀| within the unit disc

    struct sample_t {
        glm::dvec2 offset;      // β - β₀
        double recp_pdf;        // one over the density of β (in d²β)
    };

private:
    double A, B, g;
    std::array<double, tis_samples> tis_table;
    std::vector<double> edges;  // sampling_bins+1 radii
    std::vector<double> cdf;    // sampling_bins+1, from 0 to 1
    double total;               // ∫₀² 2πr BSDF(r) dr

public:
    /** @brief Throws `std::invalid_argument` unless A ≥ 0, B > 0, and the TIS at normal incidence is at most 1. */
    abg_scatter_t(double A, double B, double g);

    [[nodiscard]] inline double a() const noexcept { return A; }
    [[nodiscard]] inline double b() const noexcept { return B; }
    [[nodiscard]] inline double exponent() const noexcept { return g; }

    [[nodiscard]] inline double bsdf(double offset) const noexcept { return A / (B + std::pow(offset, g)); }
    /** @brief The TIS at incidence |β₀| (interpolated in its table). */
    [[nodiscard]] double tis(double beta0) const noexcept;
    [[nodiscard]] inline const auto& tis_values() const noexcept { return tis_table; }

    /** @brief An offset β - β₀ from three uniform numbers in [0,1). */
    [[nodiscard]] sample_t sample(double u1, double u2, double u3) const noexcept;
};

}
