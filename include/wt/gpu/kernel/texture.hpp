/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include "beam.hpp"
#include "scene.hpp"

/*
 * Textures (`kernel::texture`): bitmap lookups (`bitmap::texture2d_tmpl_t`)
 * and their spectral uplift (`texture::bitmap_t`), and materials resolved at
 * a surface point (see `kernel::material`).
 */

namespace wt::gpu::kernel {

struct vec4 {
    real x, y, z, w;
};

WT_GPU_FN vec4 operator+(vec4 a, vec4 b) { return { a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w }; }
WT_GPU_FN vec4 operator-(vec4 a, vec4 b) { return { a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w }; }
WT_GPU_FN vec4 operator-(vec4 a) { return { -a.x, -a.y, -a.z, -a.w }; }
WT_GPU_FN vec4 operator*(real s, vec4 a) { return { s*a.x, s*a.y, s*a.z, s*a.w }; }

/** @brief glm::mix: a*(1-t) + b*t */
WT_GPU_FN vec4 mix(vec4 a, vec4 b, real t) { return (1-t)*a + t*b; }

/** @brief `bitmap::wrap_coord`: a texel coordinate wrapped into [0,dim), or -1 for black or white texels. */
WT_GPU_FN i32 wrap_coord(u32 wrap, i32 coord, u32 dim) {
    const i32 d = i32(dim);
    if (coord>=0 && coord<d)
        return coord;
    if (wrap==wrap_black || wrap==wrap_white)
        return -1;
    if (wrap==wrap_clamp)
        return clamp(coord, 0, max(1,d)-1);
    const i32 period = wrap==wrap_mirror ? 2*d : d;
    i32 r = coord % period;
    if (r<0)
        r += period;
    return wrap==wrap_mirror && r>=d ? 2*d-1-r : r;
}

/**
 * @brief Component `i` (of the bitmap's, row by row) of `bm`'s texels: their
 *        stored value, decoded to a linear value (`texel_native`). `alpha`:
 *        of an alpha channel, which colour encodings leave linear.
 */
WT_GPU_FN real texel_component(const WT_GPU_THREAD plt_scene& scene, const WT_GPU_THREAD bitmap& bm, u64 i,
                               bool alpha) {
    if (bm.format==texel_float32)
        return real(float_from_bits(scene.texels[bm.offset + i]));
    const u32 lut = alpha ? bm.alpha_lut : bm.colour_lut;
    if (bm.format==texel_unorm8)
        return scene.texel_luts[lut + ((scene.texels[bm.offset + i/4] >> (8*u32(i%4))) & 0xffu)];
    return scene.texel_luts[lut + ((scene.texels[bm.offset + i/2] >> (16*u32(i%2))) & 0xffffu)];
}

/**
 * @brief `texture2d_tmpl_t::texel_native`: a texel at an image coordinate, in
 *        the bitmap's layout (its components first, the rest 0).
 */
WT_GPU_FN vec4 texel(const WT_GPU_THREAD plt_scene& scene, const WT_GPU_THREAD bitmap& bm, i32 x, i32 y) {
    x = wrap_coord(bm.uwrap, x, bm.width);
    y = wrap_coord(bm.vwrap, y, bm.height);
    if (x<0 || y<0) {
        // black or white texels (alpha 1)
        const real v = x<0 ? bm.outside_u : bm.outside_v;
        vec4 t = { v, bm.components>1 ? v : real(0), bm.components>2 ? v : real(0), bm.components>3 ? v : real(0) };
        if (bm.layout==layout_LA) t.y = 1;
        if (bm.layout==layout_RGBA) t.w = 1;
        return t;
    }

    const u64 i = (u64(y)*bm.width + u64(x))*bm.components;
    return {
        texel_component(scene, bm, i, false),
        bm.components>1 ? texel_component(scene, bm, i+1, bm.layout==layout_LA) : real(0),
        bm.components>2 ? texel_component(scene, bm, i+2, false) : real(0),
        bm.components>3 ? texel_component(scene, bm, i+3, true) : real(0),
    };
}

/** @brief `bitmap::clamp_texel` */
WT_GPU_FN real clamp_texel(u32 mode, real x) {
    return mode==clamp_non_negative ? max(real(0), x) :
           mode==clamp_non_positive ? min(real(0), x) : x;
}

/** @brief Catmull-Rom interpolation of `texture2d_tmpl_t::bicubic_native`. */
WT_GPU_FN vec4 bicubic_filter(real x, vec4 p0, vec4 p1, vec4 p2, vec4 p3) {
    return
        p1 +
        (real(.5)*x)       * (-p0 + p2) +
        (real(.5)*x*x)     * ((real(2)*p0 - real(5)*p1) + real(4)*p2 - p3) +
        (real(.5)*x*x*x)   * (((-p0 + real(3)*p1) - real(3)*p2) + p3);
}

/**
 * @brief `texture2d_tmpl_t::filter` (`bitmap_t::get_RGBA`): the bitmap
 *        filtered at `uv`, clamped, in RGBA.
 */
WT_GPU_FN vec4 filter_bitmap(const WT_GPU_THREAD plt_scene& scene, const WT_GPU_THREAD bitmap& bm, vec2 uv) {
    // flip v
    uv.y = 1-uv.y;
    const vec2 dim = { real(bm.width), real(bm.height) };
    const vec2 xy = dim*uv - vec2{ real(.5), real(.5) };

    vec4 v;
    if (bm.filter==filter_nearest) {
        v = texel(scene, bm, i32(round(xy.x)), i32(round(xy.y)));
    } else {
        const real fx = floor(xy.x), fy = floor(xy.y);
        const i32 ix = i32(fx), iy = i32(fy);
        const vec2 f = { xy.x-fx, xy.y-fy };
        if (bm.filter==filter_bilinear) {
            const vec4 t00 = texel(scene, bm, ix,   iy);
            const vec4 t01 = texel(scene, bm, ix,   iy+1);
            const vec4 t10 = texel(scene, bm, ix+1, iy);
            const vec4 t11 = texel(scene, bm, ix+1, iy+1);
            v = mix(mix(t00, t10, f.x), mix(t01, t11, f.x), f.y);
        } else {
            vec4 ts[4];
            for (i32 y=0; y<4; ++y)
                ts[y] = bicubic_filter(f.x,
                                       texel(scene, bm, ix-1, iy+y-1),
                                       texel(scene, bm, ix,   iy+y-1),
                                       texel(scene, bm, ix+1, iy+y-1),
                                       texel(scene, bm, ix+2, iy+y-1));
            v = bicubic_filter(f.y, ts[0], ts[1], ts[2], ts[3]);
        }
    }

    // clamp the bitmap's components
    v.x = clamp_texel(bm.clamp, v.x);
    if (bm.components>1) v.y = clamp_texel(bm.clamp, v.y);
    if (bm.components>2) v.z = clamp_texel(bm.clamp, v.z);
    if (bm.components>3) v.w = clamp_texel(bm.clamp, v.w);

    // convert_pixel_layout to RGBA
    if (bm.layout==layout_L)   return { v.x, v.x, v.x, 1 };
    if (bm.layout==layout_LA)  return { v.x, v.x, v.x, v.y };
    if (bm.layout==layout_RGB) return { v.x, v.y, v.z, 1 };
    return v;
}

/** @brief A texture's bitmap filtered at `uv` (`texture_t::get_RGBA`). */
WT_GPU_FN vec4 texture_rgba(const WT_GPU_THREAD plt_scene& scene, u32 index, vec2 uv) {
    const texture t = scene.textures[index];
    const bitmap bm = scene.bitmaps[t.bitmap];
    // transform_t: M*uv + translate (glm: x then y)
    const vec2 tuv = {
        t.M[0]*uv.x + t.M[1]*uv.y + t.t[0],
        t.M[2]*uv.x + t.M[3]*uv.y + t.t[1],
    };
    return filter_bitmap(scene, bm, tuv);
}

/** @brief `RGB_to_spectral::uplift` of `rgb`, with the bases' values `basis` (`spectral_sample::uplift`). */
WT_GPU_FN real uplift(vec3 rgb, const WT_GPU_THREAD real (&basis)[7]) {
    const real white = basis[0], cyan = basis[1], magenta = basis[2], yellow = basis[3];
    const real red = basis[4], green = basis[5], blue = basis[6];

    real I = 0;
    if (rgb.x <= rgb.y && rgb.x <= rgb.z) {
        I += white * rgb.x;
        if (rgb.y <= rgb.z) {
            I += cyan * (rgb.y - rgb.x);
            I += blue * (rgb.z - rgb.y);
        } else {
            I += cyan * (rgb.z - rgb.x);
            I += green * (rgb.y - rgb.z);
        }
    } else if (rgb.y <= rgb.x && rgb.y <= rgb.z) {
        I += white * rgb.y;
        if (rgb.x <= rgb.z) {
            I += magenta * (rgb.x - rgb.y);
            I += blue * (rgb.z - rgb.x);
        } else {
            I += magenta * (rgb.z - rgb.y);
            I += red * (rgb.x - rgb.z);
        }
    } else {
        I += white * rgb.z;
        if (rgb.x <= rgb.y) {
            I += yellow * (rgb.x - rgb.z);
            I += green * (rgb.y - rgb.x);
        } else {
            I += yellow * (rgb.y - rgb.z);
            I += red * (rgb.x - rgb.y);
        }
    }
    return I;
}

/**
 * @brief A texture's value at `uv` (`texture_t::f(...).x`), but for its
 *        constant factor. (Not inlined: a lookup is long, and rarely taken.)
 */
WT_GPU_NOINLINE_FN real texture_value(const WT_GPU_THREAD plt_scene& scene, u32 spectral, u32 index, vec2 uv) {
    const vec4 rgba = texture_rgba(scene, index, uv);
    if (scene.textures[index].rgb==0)
        return rgba.x;
    real basis[7];
    for (int i=0; i<7; ++i)
        basis[i] = scene.spectra[spectral].uplift[i];
    return uplift(vec3{ rgba.x, rgba.y, rgba.z }, basis);
}

/**
 * @brief The parameters of a roughness-parametrized surface profile (see
 *        `material`) of roughness `r`, at wavenumber `k` (1/m): as
 *        `fractal_t::fractal_params` and `gaussian_t::gaussian_params`.
 */
WT_GPU_FN void set_profile_roughness(WT_GPU_THREAD material& m, real r, real k) {
    // fractal_profile_details: roughness_to_T and roughness_to_alpha, matching GGX at 550 nm (lengths in mm)
    constexpr real meank = real(11423.973285781066);      // 2pi / 550nm
    constexpr real max_GGX_alpha = real(.75);
    constexpr real maxT = real(4900);                     // (70mm)^2
    const real k_mm = k * real(1e-3);
    const real a2 = sqr(clamp(r, real(0), max_GGX_alpha));
    const real T = min(maxT, (1-a2) / (4*sqr(meank)*a2));

    m.K = sqr(k_mm) * T;
    m.alpha_k = sqr(r / real(9)) * sqr(k_mm);
    if (m.profile==profile_fractal) {
        const real x = 1 + m.K;
        const real p = m.gamma==real(3) ? x : pow(x, (m.gamma-1)/2);
        m.sigma2_norm = 1/(1 - 1/p);
    } else {
        // sigma2 = 1/T
        m.s2 = 1/m.K;
        m.sigma2_norm = 1/(1 - exp(-(m.K/2)));
    }
}

/** @brief A material at spectral sample `spectral`, of textured values (see `material`), at a surface point of UV `uv`. */
WT_GPU_FN void resolve_material(const WT_GPU_THREAD plt_scene& scene, u32 spectral, WT_GPU_THREAD material& m, vec2 uv) {
    if (m.scale_texture!=no_texture)
        m.scale *= texture_value(scene, spectral, m.scale_texture, uv);
    if (m.reflectance_texture!=no_texture)
        m.reflectance = clamp(m.reflectance * texture_value(scene, spectral, m.reflectance_texture, uv),
                              real(0), real(1));
    if (m.roughness_texture!=no_texture)
        set_profile_roughness(m, m.roughness * texture_value(scene, spectral, m.roughness_texture, uv),
                              scene.spectra[spectral].k);
}

/** @brief Whether a material has textured values. */
WT_GPU_FN bool is_textured(const WT_GPU_THREAD material& m) {
    return m.scale_texture!=no_texture || m.reflectance_texture!=no_texture || m.roughness_texture!=no_texture;
}

/** @brief A normal map's shading normal at `uv`, in the unperturbed shading frame (see `texture_value`). */
WT_GPU_NOINLINE_FN vec3 normalmap_normal(const WT_GPU_THREAD plt_scene& scene, u32 index, bool flip, vec2 uv) {
    const vec4 rgba = texture_rgba(scene, index, uv);
    vec3 nm = { rgba.x*2-1, rgba.y*2-1, rgba.z*2-1 };
    if (flip) {
        nm.x = -nm.x;
        nm.y = -nm.y;
    }
    return normalize(nm);
}

/**
 * @brief The textures where a path of spectral sample `spectral` hits BVH triangle `tri` at barycentric
 *        coordinates `baryx, baryy`: the material's normal map perturbs the
 *        shading frame `shading` (of the triangle's tangent `dpdu`;
 *        `normalmap_t::shading_frame`), and its textured values resolve
 *        (`resolve_material`).
 */
WT_GPU_FN void apply_textures(const WT_GPU_THREAD plt_scene& scene, u32 spectral,
                              const WT_GPU_THREAD material_shading& ms, u32 tri, real baryx, real baryy,
                              vec3 dpdu, WT_GPU_THREAD frame& shading, WT_GPU_THREAD material& m) {
    // the interpolated UV (barycentric_t)
    const tri_uv tuv = scene.tri_uvs[tri];
    const vec2 uv = baryx*tuv.uv[0] + baryy*tuv.uv[1] + (1 - baryx - baryy)*tuv.uv[2];
    if (ms.normalmap!=no_texture)
        shading = build_shading_frame(shading.to_world(normalmap_normal(scene, ms.normalmap, ms.normalmap_flip!=0, uv)),
                                      dpdu);
    resolve_material(scene, spectral, m, uv);
}

}
