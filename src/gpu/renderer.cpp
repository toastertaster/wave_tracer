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

#include <bit>
#include <format>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <wt/ads/bvh8w/bvh8w.hpp>
#include <wt/scene/scene.hpp>
#include <wt/integrator/plt_path/plt_path.hpp>
#include <wt/integrator/plt_bdpt/plt_bdpt.hpp>
#include <wt/interaction/fsd/fraunhofer/fsd_lut.hpp>
#include <wt/emitter/area.hpp>
#include <wt/emitter/directional.hpp>
#include <wt/emitter/point.hpp>
#include <wt/emitter/spot.hpp>
#include <wt/math/util.hpp>
#include <wt/scene/shape.hpp>
#include <wt/bsdf/two_sided.hpp>
#include <wt/bsdf/composite.hpp>
#include <wt/bsdf/diffuse.hpp>
#include <wt/bsdf/scale.hpp>
#include <wt/bsdf/surface_spm.hpp>
#include <wt/bsdf/normalmap.hpp>
#include <wt/interaction/surface_profile/dirac.hpp>
#include <wt/interaction/surface_profile/fractal.hpp>
#include <wt/interaction/surface_profile/gaussian.hpp>
#include <wt/texture/constant.hpp>
#include <wt/texture/bitmap.hpp>
#include <wt/texture/scale.hpp>
#include <wt/texture/transform.hpp>
#include <wt/spectrum/colourspace/RGB/RGB_to_spectral.hpp>
#include <wt/sensor/sensor/perspective.hpp>
#include <wt/sensor/sensor/virtual_plane_sensor.hpp>
#include <wt/sensor/film/film_storage.hpp>
#include <wt/sampler/sampler.hpp>
#include <wt/scene/element/attributes.hpp>
#include <wt/util/logger/logger.hpp>
#include <wt/gpu/kernel/plt_path.hpp>
#include <wt/gpu/kernel/fraunhofer.hpp>

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

/**
 * @brief plt_bdpt's free-space diffraction: the lookup tables its sampler
 *        samples edges' diffraction lobes with, as `kernel::plt_scene::fsd_lut`.
 */
std::vector<real> compile_fsd_lut(const integrator::plt_bdpt_t& integrator) {
    namespace kf = kernel::fraunhofer;
    using lut_t = fraunhofer::fsd_sampler::fsd_lut_t;
    static_assert(lut_t::theta_samples==kf::fsd_theta_samples && lut_t::radius_samples==kf::fsd_radius_samples,
                  "the kernels' free-space diffraction lookup tables differ from the CPU's");

    const auto* sampler = integrator.get_fsd_sampler();
    if (!sampler)
        throw std::logic_error("plt_bdpt's free-space diffraction sampler was not loaded");
    const auto& lut = sampler->lut();

    std::vector<real> out;
    out.reserve(kf::fsd_lut_size);
    for (int lobe=0; lobe<2; ++lobe)
        for (const auto v : lut.iCDF_theta(lobe))
            out.push_back(real(v));
    for (int lobe=0; lobe<2; ++lobe)
        for (const auto& row : lut.iCDF_radius(lobe))
            for (const auto v : row)
                out.push_back(real(v));
    return out;
}

void compile_integrator(const scene_t& scene, const sensor::sensor_t& sensor,
                        plt_scene_t& out) {
    const auto& integrator = scene.integrator();
    auto& params = out.params;
    params = {};
    params.ray_tracing_only = sensor.ray_trace_only() ? 1 : 0;

    if (const auto* plt_path = dynamic_cast<const integrator::plt_path_t*>(&integrator); plt_path) {
        const auto& opts = plt_path->get_options();
        params.backward = opts.transport_direction==bsdf::transport_e::backward ? 1 : 0;
        params.max_depth = opts.max_depth;
        params.russian_roulette = opts.RR ? 1 : 0;
        params.fsd = opts.FSD ? 1 : 0;
        return;
    }

    if (const auto* plt_bdpt = dynamic_cast<const integrator::plt_bdpt_t*>(&integrator); plt_bdpt) {
        const auto& opts = plt_bdpt->get_options();
        params.bdpt = 1;
        params.max_depth = opts.max_depth;
        params.russian_roulette = opts.RR ? 1 : 0;
        params.mis = opts.MIS ? 1 : 0;
        params.sensor_direct = opts.sensor_direct ? 1 : 0;
        params.emitter_direct = opts.emitter_direct ? 1 : 0;
        // Fraunhofer free-space diffraction, of beams (rays never diffract)
        params.fsd = opts.FSD && !sensor.ray_trace_only() ? 1 : 0;
        if (params.fsd!=0)
            out.fsd_lut = compile_fsd_lut(*plt_bdpt);
        return;
    }

    throw gpu_error_t(std::format(
        "GPU rendering supports only the 'plt_path' and 'plt_bdpt' integrators, not '{}'", type_of(integrator)));
}

/** @brief A direction transform (as `transform_t::operator()(dir3_t)` applies it, before normalizing), row-major. */
void compile_direction_transform(const transform_t& t, real (&out)[9]) {
    const auto it = t.inverse_transpose();
    for (int c=0; c<3; ++c) {
        auto e = vec3_t{ 0 };
        e[c] = 1;
        const auto column = it(e, transform_vector);
        for (int r=0; r<3; ++r)
            out[3*r+c] = real(column[r]);
    }
}

/**
 * @brief The emitters the sensor samples (in the order of the emitter CDF,
 *        which this compiles), but for what depends on the wavenumber.
 *        `shape_emitter[i]` is the emitter of scene shape i, or no_emitter.
 */
std::vector<const emitter::emitter_t*> compile_emitters(const scene_t& scene, const sensor::sensor_t& sensor,
                                                        const ads::bvh8w_t& bvh, plt_scene_t& out,
                                                        std::vector<std::uint32_t>& shape_emitter) {
    shape_emitter.assign(scene.shapes().size(), kernel::no_emitter);
    std::vector<const emitter::emitter_t*> emitters;
    std::vector<double> pdfs;
    for (const auto& e : scene.emitters()) {
        // emitters that are never sampled for this sensor do not matter
        const auto pdf = scene.pdf_emitter(&sensor, e.get());
        if (!(pdf>0))
            continue;

        kernel::emitter k{};
        k.pmf = real(pdf);
        if (const auto* point = dynamic_cast<const emitter::point_t*>(e.get()); point) {
            k.type = kernel::emitter_point;
            k.position = to_kernel(point->get_position());
        } else if (const auto* spot = dynamic_cast<const emitter::spot_t*>(e.get()); spot) {
            k.type = kernel::emitter_spot;
            k.position = to_kernel(spot->position());
            compile_direction_transform(spot->to_world_transform(), k.dir_to_world);
            compile_direction_transform(spot->to_world_transform().inverse(), k.dir_to_local);
            const auto cutoff = f_t(u::to_rad(spot->get_cutoff()));
            const auto falloff = f_t(u::to_rad(spot->get_falloff()));
            k.cos_cutoff = real(std::cos(cutoff));
            k.cos_falloff = real(std::cos(falloff));
            k.cutoff = real(cutoff);
            k.recp_cutoff_range = real(f_t(1) / (cutoff-falloff));
            k.cutoff_solid_angle = real(m::two_pi * (1-std::cos(cutoff)));
        } else if (const auto* directional = dynamic_cast<const emitter::directional_t*>(e.get()); directional) {
            k.type = kernel::emitter_directional;
            k.target = to_kernel(directional->target_frame());
            k.target_centre = to_kernel(directional->target_centre());
            k.target_radius = real(u::to_m(directional->target_disc_radius()));
            k.far = real(u::to_m(directional->distance_to_outside()));
        } else if (const auto* area = dynamic_cast<const emitter::area_t*>(e.get()); area) {
            if (area->radiance_texture())
                throw gpu_error_t(std::format(
                    "GPU rendering supports only area emitters of a constant radiance so far; "
                    "emitter <{}> has a radiance texture", e->get_id()));
            const auto* shape = area->get_shape();
            const auto shape_it = std::ranges::find_if(scene.shapes(), [&](const auto& s) { return s.get()==shape; });
            if (shape_it==scene.shapes().end())
                throw gpu_error_t(std::format("GPU rendering found no shape of area emitter <{}>", e->get_id()));
            const auto shape_idx = std::size_t(shape_it - scene.shapes().begin());
            shape_emitter[shape_idx] = std::uint32_t(emitters.size());

            // the shape's (BVH) triangles, and their inclusive CDF by area
            k.type = kernel::emitter_area;
            k.first_tri = std::uint32_t(out.area_tris.size());
            double total = 0;
            std::vector<double> cdf;
            for (std::size_t i=0; i<bvh.triangles_count(); ++i) {
                const auto& t = bvh.tri(ads::tuid_t{ ads::idx_t(i) });
                if (t.shape_idx!=shape_idx)
                    continue;
                const auto& mt = shape->get_mesh().triangle(t.shape_tri_idx);
                total += double(u::to_m2(util::tri_surface_area(mt.p[0], mt.p[1], mt.p[2])));
                out.area_tris.push_back(std::uint32_t(i));
                cdf.push_back(total);
            }
            if (cdf.empty() || !(total>0))
                throw gpu_error_t(std::format("GPU rendering found no surface of area emitter <{}>", e->get_id()));
            for (const auto c : cdf)
                out.area_cdf.push_back(real(c/total));
            out.area_cdf.back() = 1;
            k.tri_count = std::uint32_t(cdf.size());
            k.area = real(u::to_m2(shape->get_surface_area()));
        } else {
            throw gpu_error_t(std::format(
                "GPU rendering does not support '{}' emitters yet (emitter <{}>)", type_of(*e), e->get_id()));
        }
        out.emitters.push_back(k);
        emitters.push_back(e.get());
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

/**
 * @brief The textures of the scene's materials that vary over surfaces (see
 *        `kernel::texture`): bitmaps, at transformed UVs, scaled by constant
 *        textures. Materials register theirs (`register_bsdf`) before they
 *        are compiled at any wavenumber.
 */
class textures_t {
public:
    /** @brief A texture at a wavenumber: a constant factor, times the value of `texture` (unless no_texture). */
    struct compiled_t {
        f_t factor;
        std::uint32_t texture;
    };

    /** @brief Registers the textures of `bsdf` and the BSDFs it nests, refusing those the GPU cannot look up. */
    void register_bsdf(const bsdf::bsdf_t& bsdf) {
        if (const auto* two_sided = dynamic_cast<const bsdf::two_sided_t*>(&bsdf); two_sided) {
            register_bsdf(*two_sided->nested_bsdf());
        } else if (const auto* composite = dynamic_cast<const bsdf::composite_t*>(&bsdf); composite) {
            for (const auto& nested : composite->nested_bsdfs())
                register_bsdf(*nested.second);
        } else if (const auto* scale = dynamic_cast<const bsdf::scale_t*>(&bsdf); scale) {
            register_texture(*scale->scale_texture(), bsdf);
            register_bsdf(*scale->nested_bsdf());
        } else if (const auto* normalmap = dynamic_cast<const bsdf::normalmap_t*>(&bsdf); normalmap) {
            register_normalmap(*normalmap);
            register_bsdf(*normalmap->nested_bsdf());
        } else if (const auto* diffuse = dynamic_cast<const bsdf::diffuse_t*>(&bsdf); diffuse) {
            register_texture(*diffuse->reflectance(), bsdf);
        } else if (const auto* spm = dynamic_cast<const bsdf::surface_spm_t*>(&bsdf); spm) {
            if (const auto* roughness = roughness_texture(*spm); roughness)
                register_texture(*roughness, bsdf);
        }
        // compile_bsdf refuses other BSDFs
    }

    /**
     * @brief The normal map of a material (`kernel::material_shading`): of a
     *        `normalmap_t` that the BSDF's shading frame comes from.
     */
    [[nodiscard]] kernel::material_shading shading_of(const bsdf::bsdf_t& bsdf) const {
        kernel::material_shading ret{ .normalmap = kernel::no_texture, .normalmap_flip = 0 };
        // the BSDFs that pass shading_frame on to the nested BSDF (composite_t does not)
        const bsdf::bsdf_t* b = &bsdf;
        for (;;) {
            if (const auto* two_sided = dynamic_cast<const bsdf::two_sided_t*>(b); two_sided) {
                b = two_sided->nested_bsdf().get();
            } else if (const auto* scale = dynamic_cast<const bsdf::scale_t*>(b); scale) {
                b = scale->nested_bsdf().get();
            } else if (const auto* normalmap = dynamic_cast<const bsdf::normalmap_t*>(b); normalmap) {
                if (ret.normalmap!=kernel::no_texture)
                    throw gpu_error_t(std::format(
                        "GPU rendering does not support nested normal maps (BSDF <{}>)", bsdf.get_id()));
                ret.normalmap = registered.at(normalmap->normalmap_texture().get());
                ret.normalmap_flip = normalmap->is_flipped() ? 1 : 0;
                b = normalmap->nested_bsdf().get();
            } else {
                return ret;
            }
        }
    }

    /** @brief A registered texture at wavenumber `k`. */
    [[nodiscard]] compiled_t compile(const texture::texture_t& texture, const wavenumber_t k,
                                     const bsdf::bsdf_t& of) const {
        const auto c = canonical(texture, k, of);
        return { c.factor, c.bitmap ? registered.at(&texture) : kernel::no_texture };
    }

    /** @brief The roughness texture of a roughness-parametrized fractal or gaussian surface profile, or nullptr. */
    [[nodiscard]] static const texture::texture_t* roughness_texture(const bsdf::surface_spm_t& spm) noexcept {
        const auto* profile = spm.surface_profile().get();
        if (const auto* fractal = dynamic_cast<const surface_profile::fractal_t*>(profile); fractal)
            return fractal->roughness_texture();
        if (const auto* gaussian = dynamic_cast<const surface_profile::gaussian_t*>(profile); gaussian)
            return gaussian->roughness_texture();
        return nullptr;
    }

    std::vector<kernel::texture> textures;
    std::vector<kernel::bitmap> bitmaps;
    std::vector<std::uint32_t> texels;
    std::vector<real> texel_luts;

private:
    /** @brief A texture in the form of `kernel::texture`: factor * bitmap(M*uv + t), or a constant (no bitmap). */
    struct canonical_t {
        f_t factor = 1;
        const texture::bitmap_t* bitmap = nullptr;
        mat2_t M = mat2_t{ 1 };
        vec2_t t = vec2_t{ 0 };
    };

    /** @brief `texture` in canonical form; its factor at wavenumber `k`, or 1 without. */
    static canonical_t canonical(const texture::texture_t& texture, const std::optional<wavenumber_t> k,
                                 const bsdf::bsdf_t& of) {
        if (const auto* constant = dynamic_cast<const texture::constant_t*>(&texture); constant)
            return { .factor = k ? constant->get_spectrum()->f(*k) : f_t(1) };
        if (const auto* bitmap = dynamic_cast<const texture::bitmap_t*>(&texture); bitmap)
            return { .bitmap = bitmap };
        if (const auto* transform = dynamic_cast<const texture::transform_t*>(&texture); transform) {
            // the nested texture's lookup at transform's M*uv + t
            auto c = canonical(*transform->get_nested_texture(), k, of);
            c.t = c.M*transform->get_translation() + c.t;
            c.M = c.M*transform->get_matrix();
            return c;
        }
        if (const auto* scale = dynamic_cast<const texture::scale_t*>(&texture); scale) {
            auto a = canonical(*scale->get_nested_texture(), k, of);
            auto b = canonical(*scale->get_scale_texture(), k, of);
            if (a.bitmap && b.bitmap)
                throw gpu_error_t(std::format(
                    "GPU rendering does not support scale textures of two bitmaps yet (BSDF <{}>)", of.get_id()));
            if (b.bitmap) {
                b.factor *= a.factor;
                return b;
            }
            a.factor *= b.factor;
            return a;
        }
        throw gpu_error_t(std::format(
            "GPU rendering supports only bitmap, constant, scale and transform textures so far; "
            "BSDF <{}> has a '{}' texture", of.get_id(), type_of(texture)));
    }

    void register_texture(const texture::texture_t& texture, const bsdf::bsdf_t& of) {
        if (registered.contains(&texture))
            return;
        const auto c = canonical(texture, std::nullopt, of);
        if (!c.bitmap)
            return;

        // glm matrices are column-major
        kernel::texture t{};
        t.bitmap = bitmap_index(*c.bitmap, of);
        t.rgb = c.bitmap->is_RGB() ? 1 : 0;
        t.M[0] = real(c.M[0][0]); t.M[1] = real(c.M[1][0]);
        t.M[2] = real(c.M[0][1]); t.M[3] = real(c.M[1][1]);
        t.t[0] = real(c.t.x); t.t[1] = real(c.t.y);
        registered.emplace(&texture, std::uint32_t(textures.size()));
        textures.push_back(t);
    }

    /** @brief The texture that `texture`'s transforms, if any, transform. */
    static const texture::texture_t* innermost(const texture::texture_t& texture) noexcept {
        const auto* t = &texture;
        while (const auto* transform = dynamic_cast<const texture::transform_t*>(t))
            t = transform->get_nested_texture();
        return t;
    }

    void register_normalmap(const bsdf::normalmap_t& normalmap) {
        // normalmap_t::shading_frame uses the texture's RGBA, of a bitmap (constant textures have none)
        const auto& texture = *normalmap.normalmap_texture();
        const auto c = canonical(texture, std::nullopt, normalmap);
        if (!c.bitmap || !dynamic_cast<const texture::bitmap_t*>(innermost(texture)))
            throw gpu_error_t(std::format(
                "GPU rendering supports only bitmap (and transform) normal map textures (BSDF <{}>)",
                normalmap.get_id()));
        register_texture(texture, normalmap);
    }

    /**
     * @brief A bitmap, its texels as the CPU stores them (see `kernel::bitmap`): unorm integers, whose
     *        decoding tables it shares with other bitmaps of their width and colour encoding, or floats.
     */
    std::uint32_t bitmap_index(const texture::bitmap_t& bitmap, const bsdf::bsdf_t& of) {
        const auto& tex = *bitmap.get_texture();
        if (const auto it = bitmap_of.find(&tex); it!=bitmap_of.end())
            return it->second;

        using bitmap::pixel_layout_e;
        const auto layout = tex.pixel_layout();
        if (layout==pixel_layout_e::Custom)
            throw gpu_error_t(std::format(
                "GPU rendering does not support bitmaps of a custom pixel layout (BSDF <{}>)", of.get_id()));
        const auto dim = tex.dimensions();
        const std::uint32_t components = tex.components();
        const std::uint64_t count = std::uint64_t(dim.x)*dim.y*components;

        // the texture2d_loader's texel types: unorm8 and unorm16 (PNG), f_t (EXR)
        const auto component_bytes = tex.component_bytes();
        const auto format = component_bytes==1 ? kernel::texel_unorm8 :
                            component_bytes==2 ? kernel::texel_unorm16 : kernel::texel_float32;
        const std::uint32_t per_word = format==kernel::texel_unorm8 ? 4 : format==kernel::texel_unorm16 ? 2 : 1;
        const std::uint64_t words = (count + per_word-1) / per_word;
        if (texels.size()+words > std::uint64_t(1)<<32)
            throw gpu_error_t("GPU rendering supports bitmaps of up to 16 GiB in all");

        const auto& config = tex.get_config();
        kernel::bitmap bm{};
        bm.offset = std::uint32_t(texels.size());
        bm.format = format;
        bm.width = dim.x;
        bm.height = dim.y;
        bm.layout = layout==pixel_layout_e::L ? kernel::layout_L :
                    layout==pixel_layout_e::LA ? kernel::layout_LA :
                    layout==pixel_layout_e::RGB ? kernel::layout_RGB : kernel::layout_RGBA;
        bm.components = components;
        bm.filter = std::uint32_t(config.filter);
        bm.uwrap = std::uint32_t(config.uwrap);
        bm.vwrap = std::uint32_t(config.vwrap);
        bm.clamp = std::uint32_t(config.texel_clamp_mode);
        // the black or white texels outside the bitmap (texel() returns them in RGBA)
        using bitmap::texture_wrap_mode_e;
        const auto outside = [](texture_wrap_mode_e wrap) {
            return wrap==texture_wrap_mode_e::black || wrap==texture_wrap_mode_e::white;
        };
        bm.outside_u = outside(config.uwrap) ? real(tex.texel({ -1,0 }).x) : real(0);
        bm.outside_v = outside(config.vwrap) ? real(tex.texel({ 0,-1 }).x) : real(0);

        // the texels, row by row, from the storage's tiles (`texture2d_tmpl_t::texel_native`), each
        // component in its (lowest unused) bits of a word
        const auto& storage = tex.get_storage();
        const auto pack = [&]<typename T>(const auto encode) {
            const auto* data = reinterpret_cast<const T*>(storage.data.get());
            texels.resize(texels.size()+words, 0);
            std::uint64_t i = 0;
            for (std::uint32_t y=0; y<dim.y; ++y)
            for (std::uint32_t x=0; x<dim.x; ++x)
            for (std::uint32_t c=0; c<components; ++c, ++i) {
                const T v = data[std::size_t(storage.texel_offset({ x,y }))*components + c];
                texels[bm.offset + i/per_word] |= encode(v) << (32/per_word * (i%per_word));
            }
        };
        if (format==kernel::texel_unorm8) {
            pack.template operator()<std::uint8_t>([](std::uint8_t v) { return std::uint32_t(v); });
            bm.colour_lut = decoding_lut<std::uint8_t>(tex.colour_encoding());
            bm.alpha_lut = decoding_lut<std::uint8_t>(bitmap::colour_encoding_type_e::linear);
        } else if (format==kernel::texel_unorm16) {
            pack.template operator()<std::uint16_t>([](std::uint16_t v) { return std::uint32_t(v); });
            bm.colour_lut = decoding_lut<std::uint16_t>(tex.colour_encoding());
            bm.alpha_lut = decoding_lut<std::uint16_t>(bitmap::colour_encoding_type_e::linear);
        } else {
            pack.template operator()<f_t>([](f_t v) { return std::bit_cast<std::uint32_t>(float(v)); });
        }

        const auto index = std::uint32_t(bitmaps.size());
        bitmaps.push_back(bm);
        bitmap_of.emplace(&tex, index);
        return index;
    }

    /**
     * @brief The table of the linear values of unorm texels of type `T` in colour encoding `ce`
     *        (`colour_encoding_t::to_linear_fp`; linear: `to_fp`, as of alpha): its offset in `texel_luts`.
     */
    template <typename T>
    std::uint32_t decoding_lut(const bitmap::colour_encoding_t ce) {
        const auto key = std::make_pair(sizeof(T), ce.gamma);
        if (const auto it = luts.find(key); it!=luts.end())
            return it->second;
        const auto offset = std::uint32_t(texel_luts.size());
        for (std::uint32_t i=0; i<=std::numeric_limits<T>::max(); ++i)
            texel_luts.push_back(real(ce.to_linear_fp(T(i))));
        luts.emplace(key, offset);
        return offset;
    }

    std::map<const texture::texture_t*, std::uint32_t> registered;
    std::map<const bitmap::texture2d_t*, std::uint32_t> bitmap_of;
    std::map<std::pair<std::size_t, f_t>, std::uint32_t> luts;     // decoding tables, by texel size and gamma
};

/** @brief The surface profile of a `surface_spm_t` at wavenumber `k` (see `kernel::material`). */
void compile_surface_profile(const bsdf::surface_spm_t& spm, const wavenumber_t k, const textures_t& textures,
                             kernel::material& m) {
    const auto& profile = *spm.surface_profile();
    if (dynamic_cast<const surface_profile::dirac_t*>(&profile)) {
        m.profile = kernel::profile_dirac;
        return;
    }

    const auto* fractal = dynamic_cast<const surface_profile::fractal_t*>(&profile);
    const auto* gaussian = dynamic_cast<const surface_profile::gaussian_t*>(&profile);
    if (!fractal && !gaussian)
        throw gpu_error_t(std::format(
            "GPU rendering does not support '{}' surface profiles yet (BSDF <{}>)", type_of(profile), spm.get_id()));
    m.profile = fractal ? kernel::profile_fractal : kernel::profile_gaussian;
    if (fractal)
        m.gamma = real(fractal->get_gamma());

    // a textured roughness: the kernel derives the parameters where paths hit (`set_profile_roughness`)
    if (const auto* roughness = textures_t::roughness_texture(spm); roughness) {
        const auto r = textures.compile(*roughness, k, spm);
        if (r.texture!=kernel::no_texture) {
            m.roughness = real(r.factor);
            m.roughness_texture = r.texture;
            return;
        }
    } else if (fractal ? !fractal->has_constant_textures() : !gaussian->has_constant_textures()) {
        throw gpu_error_t(std::format(
            "GPU rendering supports textured surface profiles only of a (perceptual) roughness; BSDF <{}> has a "
            "textured '{}' surface profile", spm.get_id(), type_of(profile)));
    }

    // constant textures: any query will do
    const auto query = texture::texture_query_t{ .uv = {}, .k = k };
    const auto k_mm = f_t(k * u::mm);
    if (fractal) {
        const auto params = fractal->parameters(query);
        m.K = real(u::to_num(k*k * params.T));
        m.sigma2_norm = real(params.sigma2_norm);
        m.alpha_k = real(params.alpha * m::sqr(k_mm));
    } else {
        const auto params = gaussian->parameters(query);
        m.s2 = real(u::to_num(params.sigma2 / (k*k)));
        m.sigma2_norm = real(params.sigma2_norm);
        m.alpha_k = real(params.alpha * m::sqr(k_mm));
    }
}

kernel::material compile_bsdf(const bsdf::bsdf_t& bsdf, const wavenumber_t k, const textures_t& textures) {
    kernel::material m{};
    m.type = kernel::material_absorbing;
    m.scale = 1;
    m.scale_texture = m.roughness_texture = m.reflectance_texture = kernel::no_texture;

    // unwrap the BSDFs that select, reorient or scale another (normal maps perturb the shading frame instead,
    // see `textures_t::shading_of`)
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
            const auto s = textures.compile(*scale->scale_texture(), k, *scale);
            m.scale *= real(s.factor);
            if (s.texture!=kernel::no_texture) {
                if (m.scale_texture!=kernel::no_texture)
                    throw gpu_error_t(std::format(
                        "GPU rendering supports one textured scale BSDF per material so far (BSDF <{}>)",
                        bsdf.get_id()));
                m.scale_texture = s.texture;
            }
            b = scale->nested_bsdf().get();
        } else if (const auto* normalmap = dynamic_cast<const bsdf::normalmap_t*>(b); normalmap) {
            b = normalmap->nested_bsdf().get();
        } else {
            break;
        }
    }

    m.delta_only = b->is_delta_only(k) ? 1 : 0;

    if (const auto* diffuse = dynamic_cast<const bsdf::diffuse_t*>(b); diffuse) {
        m.type = kernel::material_diffuse;
        const auto r = textures.compile(*diffuse->reflectance(), k, *diffuse);
        // diffuse_t clamps the reflectance; the kernel clamps a textured one where paths hit
        m.reflectance_texture = r.texture;
        m.reflectance = real(r.texture==kernel::no_texture ? m::clamp01<f_t>(r.factor) : r.factor);
        return m;
    }

    if (const auto* spm = dynamic_cast<const bsdf::surface_spm_t*>(b); spm) {
        const auto eta = spm->IOR(k);
        m.type = kernel::material_specular;
        m.has_transmission = ior_has_transmission(eta) ? 1 : 0;
        m.eta = { real(eta.real()), real(eta.imag()) };
        m.reflection_scale = real(spm->reflectivity_scale(k));
        m.transmission_scale = real(spm->transmissivity_scale(k));
        compile_surface_profile(*spm, k, textures, m);
        return m;
    }

    throw gpu_error_t(std::format(
        "GPU rendering does not support '{}' BSDFs yet (BSDF <{}>)", type_of(*b), b->get_id()));
}

/**
 * @brief Compiles the geometry, with the emitters of shapes `shape_emitter`,
 *        and the materials' textures into `textures`; returns the materials'
 *        BSDFs (compiled for each spectral sample).
 */
std::vector<const bsdf::bsdf_t*> compile_geometry(const scene_t& scene, const ads::bvh8w_t& bvh,
                                                  const std::vector<std::uint32_t>& shape_emitter,
                                                  textures_t& textures, plt_scene_t& out) {
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

    // the textures, and the triangles' UVs if any
    for (const auto* bsdf : bsdfs)
        textures.register_bsdf(*bsdf);
    for (const auto* bsdf : bsdfs)
        out.material_shadings.push_back(textures.shading_of(*bsdf));
    if (!textures.textures.empty()) {
        out.tri_uvs.resize(bvh.triangles_count());
        for (std::size_t i=0; i<out.tri_uvs.size(); ++i) {
            const auto& t = bvh.tri(ads::tuid_t{ ads::idx_t(i) });
            const auto& mesh_tri = scene.shapes()[t.shape_idx]->get_mesh().triangle(t.shape_tri_idx);
            for (int v=0; v<3; ++v) {
                const auto uv = mesh_tri.uv ? (*mesh_tri.uv)[v] : vec2_t{ 0 };
                out.tri_uvs[i].uv[v] = { real(uv.x), real(uv.y) };
            }
        }
    }
    out.textures = textures.textures;
    out.bitmaps = textures.bitmaps;
    out.texels = std::move(textures.texels);
    out.texel_luts = std::move(textures.texel_luts);

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
        s.emitter = shape_emitter[t.shape_idx];
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
    const sensor::response::response_t* response;
    std::uint32_t components;
    std::function<beam::sourcing_geometry_t(wavenumber_t)> sourcing;
};

template <std::size_t Dims, bool polarimetric>
void compile_film(const sensor::film_t<Dims, polarimetric>& film, plt_scene_t& out) {
    const auto size = film.dimensions();
    const auto components = film.pixel_layout().components;
    if (components>4)
        throw gpu_error_t("GPU rendering supports films of up to 4 components");
    if (std::uint64_t(size.x)*size.y*components*(polarimetric ? 4 : 1) > std::uint64_t(1)<<32)
        throw gpu_error_t("GPU rendering supports films of up to 2^32 values");
    if (film.reconstruction_filter_radius()>kernel::max_rf_radius)
        throw gpu_error_t(std::format(
            "GPU rendering supports film reconstruction filters up to {} elements wide; "
            "reduce the film's rfilter_scale", kernel::max_rf_radius));

    auto& fd = out.film;
    fd.width = size.x;
    fd.height = size.y;
    fd.components = components;
    fd.polarimetric = polarimetric ? 1 : 0;
    fd.rf_radius = kernel::i32(film.reconstruction_filter_radius());
    fd.rf_sigma = real(film.reconstruction_filter_stddev());
}

/** @brief A row-major 4x4 matrix of a transform (glm matrices are column-major). */
void compile_matrix(const transform_t& t, real (&out)[16]) {
    const auto& m = t.matrix();
    for (int r=0; r<4; ++r)
    for (int c=0; c<4; ++c)
        out[4*r+c] = real(m[c][r]);
}

sensor_spectral_t compile_sensor(const sensor::sensor_t& sensor, const kernel::plt_params& params,
                                 plt_scene_t& out) {
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
        return { plane->film().response(), plane->film().pixel_layout().components,
                 [plane](wavenumber_t k) { return plane->sourcing_geometry(k); } };
    }

    // perspective_t::sample, sample_direct and the densities; forward plt_path cannot reach a point sensor
    const auto perspective = [&]<bool polarimetric>(const sensor::perspective_t<polarimetric>& perspective) {
        if (!params.sensor_paths())
            throw gpu_error_t(std::format(
                "GPU rendering supports perspective sensors only with backward plt_path "
                "(direction=\"backward\") or plt_bdpt; sensor <{}>", sensor.get_id()));
        s.type = kernel::sensor_perspective;
        s.position = to_kernel(perspective.position());
        compile_matrix(perspective.film_to_camera_transform(), s.to_camera);
        compile_matrix(perspective.film_to_camera_transform().inverse(), s.to_film);
        compile_direction_transform(perspective.to_world_transform(), s.dir_to_world);
        compile_direction_transform(perspective.to_world_transform().inverse(), s.dir_to_local);
        s.ddir_dx = to_kernel(perspective.direction_pdvx());
        s.ddir_dy = to_kernel(perspective.direction_pdvy());
        s.recp_sa_scale = real(u::to_num(perspective.recp_solid_angle_density_for_sample(dir3_t{ 0,0,1 }) /
                                         u::ang::sr));
        compile_film(perspective.film(), out);
        return sensor_spectral_t{ perspective.film().response(), perspective.film().pixel_layout().components,
                                  [p = &perspective](wavenumber_t k) { return p->sourcing_geometry(k); } };
    };
    if (const auto* p = dynamic_cast<const sensor::perspective_t<false>*>(&sensor); p)
        return perspective(*p);
    if (const auto* p = dynamic_cast<const sensor::perspective_t<true>*>(&sensor); p)
        return perspective(*p);

    throw gpu_error_t(std::format(
        "GPU rendering supports only virtual_plane and perspective sensors so far; "
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

    spectral_tables_t(const scene_t& scene, const sensor::sensor_t& sensor, const kernel::plt_params& params,
                      sensor_spectral_t sensor_spectral,
                      std::vector<const emitter::emitter_t*> emitters,
                      std::vector<const bsdf::bsdf_t*> bsdfs,
                      textures_t textures)
        : scene(scene), sensor(sensor), params(params),
          sensor_spectral(std::move(sensor_spectral)),
          emitters(std::move(emitters)), bsdfs(std::move(bsdfs)), textures(std::move(textures))
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
        current.emissions.clear();
        current.materials.clear();
        seeded_sampler_t sampler{ mix_seed(pass) };
        for (const auto* emitter : emitters)
        for (std::uint32_t j=0; j<per_emitter; ++j) {
            sampler.stratify(j, per_emitter);
            compile(scene.sample_spectrum(&sensor, emitter, sampler), scene.pdf_emitter(&sensor, emitter));
        }
        built_for = pass;
    }

    /** @brief Appends the spectral sample of wavenumber sample `ws`, drawn for an emitter of probability `emitter_pdf`. */
    void compile(const wavenumber_sample_t& ws, const f_t emitter_pdf) {
        const auto k = ws.k;
        const bool sensor_paths = params.sensor_paths();

        // spectral (importance) sampling weight: plt_path::integrate_forward (one over the probability of
        // the emitter and wavenumber sampled), integrate_backward and plt_bdpt (of the wavenumber, over all
        // emitters)
        const auto recp_wpd = ws.wpd.is_discrete() ?
            f_t(1) / wavenumber_density_t{ ws.wpd.mass() * u::mm } :
            f_t(1) / ws.wpd.density();
        const wavenumber_t recp_spectral_pd =
            !sensor_paths ? recp_wpd / emitter_pdf :
            ws.wpd.is_discrete() ? recp_wpd :
            f_t(1) / scene.sum_spectral_pdf_for_all_emitters(&sensor, k);
        // a wavenumber no emitter emits at contributes nothing
        const auto recp = recp_spectral_pd>zero && m::isfinite(recp_spectral_pd) ?
            recp_spectral_pd : wavenumber_t::zero();

        kernel::spectral_sample s{};
        s.k = real(u::to_inv_m(k));
        if (params.bdpt!=0 && recp>zero) {
            // plt_bdpt without MIS weighs by one over the wavenumber's density (of its emitter), rather than `recp`
            const auto k_density = ws.wpd.is_discrete() ?
                wavenumber_density_t{ ws.wpd.mass() / wavenumber_t::unit } :
                ws.wpd.density();
            s.no_mis_scale = real(u::to_num(f_t(1) / (k_density * recp)));
        }
        if (sensor_paths) {
            // the beams the sensor sources, which must be isotropic cones
            const auto sourcing = sensor_spectral.sourcing(k);
            if (sourcing.surface || sourcing.initial_spatial_lengths.x!=sourcing.initial_spatial_lengths.y)
                throw gpu_error_t(std::format(
                    "GPU rendering supports only sensors that source isotropic beams; sensor <{}> does not",
                    sensor.get_id()));
            s.sensor_tan_alpha = real(sourcing.tan_alpha);
            s.sensor_x0 = real(u::to_m(sourcing.initial_spatial_lengths.x));
        }
        for (std::uint32_t c=0; c<4; ++c)
            s.response.c[c] = c<sensor_spectral.components ? real(sensor_spectral.response->f(c, k)) : real(0);
        // RGB uplift: the white, cyan, magenta, yellow, red, green and blue bases
        const auto lambda = wavenum_to_wavelen(k);
        const vec3_t bases[7] = { { 1,1,1 }, { 0,1,1 }, { 1,0,1 }, { 1,1,0 }, { 1,0,0 }, { 0,1,0 }, { 0,0,1 } };
        for (int i=0; i<7; ++i)
            s.uplift[i] = real(colourspace::RGB_to_spectral::uplift(bases[i], lambda));
        current.samples.push_back(s);

        for (const auto* e : emitters)
            current.emissions.push_back(compile_emission(*e, k, recp));

        for (const auto* bsdf : bsdfs)
            current.materials.push_back(compile_bsdf(*bsdf, k, textures));
    }

    /** @brief An emitter's `kernel::emission` at wavenumber `k`, times the spectral sampling weight `recp`. */
    static kernel::emission compile_emission(const emitter::emitter_t& e, const wavenumber_t k,
                                             const wavenumber_t recp) {
        kernel::emission ret{};
        const auto sourced = [&](const beam::sourcing_geometry_t& sourcing) {
            if (sourcing.surface || sourcing.initial_spatial_lengths.x!=sourcing.initial_spatial_lengths.y)
                throw gpu_error_t(std::format(
                    "GPU rendering supports only emitters that source isotropic beams; emitter <{}> does not",
                    e.get_id()));
            ret.tan_alpha = real(sourcing.tan_alpha);
            ret.x0 = real(u::to_m(sourcing.initial_spatial_lengths.x));
        };

        // `flux`: emitter_t::sample's beam over its sampling densities; `intensity`: what sample_direct emits
        if (const auto* point = dynamic_cast<const emitter::point_t*>(&e); point) {
            sourced(point->sourcing_geometry(k));
            const auto I = point->spectral_radiant_intensity(k) * recp;
            ret.flux = real(u::to_W(I * (m::four_pi * u::ang::sr)));
            ret.intensity = real(u::to_W(I * u::ang::sr));
        } else if (const auto* spot = dynamic_cast<const emitter::spot_t*>(&e); spot) {
            sourced(spot->sourcing_geometry(k));
            const auto I = spot->spectral_radiant_intensity(k) * recp;
            const auto cutoff_solid_angle = m::two_pi * (1-std::cos(f_t(u::to_rad(spot->get_cutoff()))));
            ret.flux = real(u::to_W(I * (cutoff_solid_angle * u::ang::sr)));
            ret.intensity = real(u::to_W(I * u::ang::sr));
        } else if (const auto* directional = dynamic_cast<const emitter::directional_t*>(&e); directional) {
            sourced(directional->sourcing_geometry(k));
            const auto E = directional->spectral_irradiance(k) * recp;
            ret.flux = real(u::numerical_value_in(E * directional->target_area(), u::W));
            ret.intensity = real(u::numerical_value_in(E * square(u::m), u::W));
        } else if (const auto* area = dynamic_cast<const emitter::area_t*>(&e); area) {
            sourced(area->sourcing_geometry(k));
            const auto L = area->get_scale() * area->emission_spectrum().f(k) * spectral_radiance_t::unit * recp;
            ret.flux = real(u::numerical_value_in(L * area->get_shape()->get_surface_area() * (m::pi * u::ang::sr), u::W));
            ret.intensity = real(u::numerical_value_in(L * square(u::m) * u::ang::sr, u::W));
        }
        return ret;
    }

    const scene_t& scene;
    const sensor::sensor_t& sensor;
    kernel::plt_params params;
    sensor_spectral_t sensor_spectral;
    std::vector<const emitter::emitter_t*> emitters;
    std::vector<const bsdf::bsdf_t*> bsdfs;
    textures_t textures;

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

    plt_scene_t compiled;
    compile_integrator(scene, sensor, compiled);
    params = compiled.params;
    auto sensor_spectral = compile_sensor(sensor, params, compiled);
    std::vector<std::uint32_t> shape_emitter;
    auto emitters = compile_emitters(scene, sensor, *bvh, compiled, shape_emitter);
    textures_t textures;
    auto bsdfs = compile_geometry(scene, *bvh, shape_emitter, textures, compiled);
    // the first pass's table: compiles the BSDFs, refusing those the GPU does not support
    spectra = std::make_unique<spectral_tables_t>(scene, sensor, params, std::move(sensor_spectral),
                                                  std::move(emitters), std::move(bsdfs), std::move(textures));
    compiled.samples_per_emitter = spectra->samples_per_emitter();

    paths_per_spe = std::uint64_t(compiled.film.width) * compiled.film.height;
    film_elements = std::size_t(paths_per_spe) * compiled.film.components;
    polarimetric = compiled.film.polarimetric!=0;

    geometry = device.upload(ads);
    try {
        backend = geometry->backend->create_plt_renderer(compiled);
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
    const auto write = [&](auto* storage) {
        if (!storage || storage->pixel_layout().components*paths_per_spe!=film_elements)
            throw gpu_error_t("GPU rendering expected the sensor's 2D film");

        // as the CPU's integrators write: backward plt_path, block splats; forward, direct splats; plt_bdpt,
        // both (the direct splats into a light image of their own)
        std::vector<double> values(film_elements * storage->values_per_component);
        if (params.block_splats()) {
            std::vector<double> weights(film_elements);
            backend->read_film(values.data(), weights.data());
            storage->set_image(values.data(), weights.data());
            if (params.bdpt!=0) {
                backend->read_light_film(values.data());
                storage->set_light_image(values.data());
            }
        } else {
            backend->read_film(values.data(), nullptr);
            storage->set_light_image(values.data());
        }
    };
    if (polarimetric)
        write(dynamic_cast<sensor::film_storage_t<true, 2>*>(&film));
    else
        write(dynamic_cast<sensor::film_storage_t<false, 2>*>(&film));
}
