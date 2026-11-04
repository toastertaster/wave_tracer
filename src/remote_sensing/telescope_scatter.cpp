/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#include <algorithm>
#include <format>
#include <numbers>
#include <stdexcept>

#include <wt/remote_sensing/telescope_scatter.hpp>

using namespace wt::remote_sensing::optics;

namespace {

constexpr double pi = std::numbers::pi;
// Simpson's rule on each sampling bin, of this many (even) intervals
constexpr int simpson_intervals = 32;

// the fraction of the circle of radius r about a point at distance b0 from the origin that is within the unit disc
double inside_fraction(double r, double b0) noexcept {
    if (r<=1-b0)
        return 1;
    if (r>=1+b0)
        return 0;
    const auto c = (b0*b0 + r*r - 1) / (2*b0*r);
    return std::acos(std::clamp(c, -1., 1.)) / pi;
}

}


abg_scatter_t::abg_scatter_t(double A, double B, double g)
    : A(A), B(B), g(g)
{
    if (!(A>=0) || !(B>0) || !std::isfinite(g))
        throw std::invalid_argument("ABg scatter needs A ≥ 0, B > 0 and a finite g");

    // log-spaced bins, from well below the BSDF's knee (r = B^(1/g)), the first from 0
    const auto knee = std::pow(B, 1/std::max(g, .1));
    const auto r_lo = std::clamp(1e-3*knee, 1e-12, 1e-3);
    edges.resize(sampling_bins+1);
    edges[0] = 0;
    for (std::size_t i=1; i<=sampling_bins; ++i)
        edges[i] = r_lo * std::pow(max_offset/r_lo, double(i-1)/double(sampling_bins-1));

    // ∫ 2πr BSDF(r) f(r) dr over a bin, by Simpson's rule
    const auto integrate = [&](std::size_t bin, auto&& f) {
        const auto a = edges[bin], b = edges[bin+1];
        const auto h = (b-a) / simpson_intervals;
        double sum = 0;
        for (int j=0; j<=simpson_intervals; ++j) {
            const auto r = a + j*h;
            const auto w = j==0 || j==simpson_intervals ? 1 : j%2==1 ? 4 : 2;
            sum += w * 2*pi*r*bsdf(r) * f(r);
        }
        return sum * h/3;
    };

    cdf.resize(sampling_bins+1);
    cdf[0] = 0;
    for (std::size_t i=0; i<sampling_bins; ++i)
        cdf[i+1] = cdf[i] + integrate(i, [](double) { return 1.; });
    total = cdf.back();
    for (auto& c : cdf)
        c = total>0 ? c/total : 0;

    for (std::size_t k=0; k<tis_samples; ++k) {
        const auto b0 = double(k) / double(tis_samples-1);
        double tis = 0;
        for (std::size_t i=0; i<sampling_bins && edges[i]<1+b0; ++i)
            tis += integrate(i, [b0](double r) { return inside_fraction(r, b0); });
        tis_table[k] = tis;
    }
    if (tis_table[0]>1)
        throw std::invalid_argument(std::format("ABg scatter of total integrated scatter {:.4g} > 1 at normal incidence", tis_table[0]));
}

double abg_scatter_t::tis(double beta0) const noexcept {
    const auto x = std::clamp(beta0, 0., 1.) * double(tis_samples-1);
    const auto i = std::min(std::size_t(x), tis_samples-2);
    const auto f = x - double(i);
    return (1-f)*tis_table[i] + f*tis_table[i+1];
}

abg_scatter_t::sample_t abg_scatter_t::sample(double u1, double u2, double u3) const noexcept {
    if (!(total>0))
        return { { 0,0 }, 0 };
    // a bin by its mass, then a radius uniformly by area within it
    const auto it = std::upper_bound(cdf.begin()+1, cdf.end(), u1);
    const auto bin = std::min<std::size_t>(std::size_t(it - cdf.begin()) - 1, sampling_bins-1);
    const auto a = edges[bin], b = edges[bin+1];
    const auto r = std::sqrt(a*a + u2*(b*b - a*a));
    const auto phi = 2*pi*u3;
    const auto mass = cdf[bin+1] - cdf[bin];
    const auto bin_area = pi*(b*b - a*a);
    return {
        .offset = r * glm::dvec2{ std::cos(phi), std::sin(phi) },
        .recp_pdf = mass>0 ? bin_area / mass : 0,
    };
}
