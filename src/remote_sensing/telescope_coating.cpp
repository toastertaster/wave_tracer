/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#include <algorithm>
#include <complex>
#include <format>
#include <numbers>

#include <wt/remote_sensing/telescope_coating.hpp>
#include <wt/interaction/fresnel.hpp>

#include <wt/scene/loader/node_readers.hpp>
#include <wt/util/logger/logger.hpp>
#include "prescription.hpp"

using namespace wt;
using namespace wt::remote_sensing::optics;

namespace {

using cd = std::complex<double>;

// a characteristic matrix (Macleod's convention: refractive indices n - ik, phase e^{i(ωt - kz)})
struct matrix2_t {
    cd a, b, c, d;

    [[nodiscard]] matrix2_t operator*(const matrix2_t& o) const noexcept {
        return { a*o.a + b*o.c, a*o.b + b*o.d,
                 c*o.a + d*o.c, c*o.b + d*o.d };
    }
};

// cos θ in a medium of (Macleod) refractive index N, for incidence from vacuum at sin θ₀: the root whose
// N cos θ has a non-positive imaginary part (a wave that decays into an absorbing medium)
cd cos_in(cd N, double sin0) noexcept {
    const auto s = sin0/N;
    auto c = std::sqrt(1. - s*s);
    if (std::imag(N*c)>0)
        c = -c;
    return c;
}

// the reflection coefficient of a stack of characteristic matrix M on a substrate of admittance eta_sub, from
// vacuum of admittance eta0
cd reflect(const matrix2_t& M, cd eta0, cd eta_sub) noexcept {
    const auto B = M.a + M.b*eta_sub;
    const auto C = M.c + M.d*eta_sub;
    return (eta0*B - C) / (eta0*B + C);
}

cd macleod_ior(const spectrum::spectrum_t& s, wavenumber_t k) noexcept {
    // wave_tracer's n + ik is Macleod's n - ik
    return std::conj(cd{ s.value(k) });
}

}


fresnel_coefficients_t coating_t::reflection(wavenumber_t k, double cos_theta) const noexcept {
    if (is_ideal())
        return { .rs = c_t{ -1,0 }, .rp = c_t{ 1,0 } };

    cos_theta = std::clamp(cos_theta, 0., 1.);
    const auto sin0 = std::sqrt(1 - cos_theta*cos_theta);

    if (stack.empty()) {
        // a bare substrate: wave_tracer's own Fresnel reflection (its refractive-index ratio is outside over inside)
        const auto f = ::wt::fresnel_reflection(c_t{ 1,0 } / substrate_ior->value(k),
                                                dir3_t{ f_t(sin0), 0, f_t(cos_theta) });
        return { .rs = f.rs, .rp = f.rp };
    }

    const auto lambda = double(u::to_m(wavenum_to_wavelen(k)));

    // the layers' characteristic matrices, for s and p, from the outermost
    auto Ms = matrix2_t{ 1,0,0,1 }, Mp = matrix2_t{ 1,0,0,1 };
    for (const auto& l : stack) {
        const auto N = macleod_ior(*l.ior, k);
        const auto c = cos_in(N, sin0);
        const auto delta = 2*std::numbers::pi/lambda * N*l.thickness*c;
        const auto cosd = std::cos(delta), isind = cd{ 0,1 }*std::sin(delta);
        const auto eta_s = N*c, eta_p = N/c;
        Ms = Ms * matrix2_t{ cosd, isind/eta_s, isind*eta_s, cosd };
        Mp = Mp * matrix2_t{ cosd, isind/eta_p, isind*eta_p, cosd };
    }
    const auto Ns = macleod_ior(*substrate_ior, k);
    const auto cs = cos_in(Ns, sin0);
    const auto rs = reflect(Ms, cd{ cos_theta }, Ns*cs);
    const auto rp = reflect(Mp, cd{ 1/cos_theta }, Ns/cs);

    // back to wave_tracer's convention: conjugated (its n + ik), and its rp's sign (rp = -rs at normal incidence)
    return {
        .rs = c_t{ std::conj(rs) },
        .rp = c_t{ -std::conj(rp) },
    };
}

double coating_t::normal_reflectance(wavenumber_t k) const noexcept {
    const auto r = reflection(k, 1);
    return (std::norm(cd{ r.rs }) + std::norm(cd{ r.rp })) / 2;
}

std::string coating_t::description() const {
    if (is_ideal())
        return "ideal";
    return std::format("multilayer: {} layer(s) on a substrate <{}>", stack.size(), substrate_ior->get_id());
}

std::shared_ptr<const coating_t> coating_t::load(scene::loader::loader_t* loader,
                                                 const scene::loader::node_t& node,
                                                 const wt::wt_context_t &context) {
    const auto& type = node["type"];
    if (type=="ideal") {
        for (auto& item : node.children_view())
            throw scene_loading_exception_t("(telescope coating loader) an ideal coating has no parameters", item);
        return std::make_shared<coating_t>();
    }
    if (type!="multilayer")
        throw scene_loading_exception_t("(telescope coating loader) unsupported coating type \"" + type + "\" (supported: \"ideal\", \"multilayer\")", node);

    std::shared_ptr<spectrum::spectrum_t> substrate;
    std::vector<layer_t> layers;
    for (auto& item : node.children_view()) {
    try {
        if (item.name()=="layer") {
            std::shared_ptr<spectrum::spectrum_t> ior;
            std::optional<double> thickness;
            for (auto& l : item.children_view()) {
                if (!scene::loader::load_scene_element(l, "IOR", ior, loader, context) &&
                    !prescription::read_length(l, "thickness", thickness))
                    logger::cwarn()
                        << loader->node_description(l)
                        << "(telescope coating loader) unqueried node type " << l.name() << " (\"" << l["name"] << "\")" << '\n';
            }
            if (!ior || !thickness || !(*thickness>=0))
                throw scene_loading_exception_t("(telescope coating loader) a layer needs an 'IOR' spectrum and a non-negative 'thickness'", item);
            layers.push_back({ .ior = std::move(ior), .thickness = *thickness });
        }
        else if (!scene::loader::load_scene_element(item, "substrate", substrate, loader, context))
            logger::cwarn()
                << loader->node_description(item)
                << "(telescope coating loader) unqueried node type " << item.name() << " (\"" << item["name"] << "\")" << '\n';
    } catch(const std::format_error& exp) {
        throw scene_loading_exception_t("(telescope coating loader) " + std::string{ exp.what() }, item);
    }
    }
    if (!substrate)
        throw scene_loading_exception_t("(telescope coating loader) a multilayer coating needs a 'substrate' spectrum (its refractive index)", node);

    return std::make_shared<coating_t>(std::move(substrate), std::move(layers));
}
