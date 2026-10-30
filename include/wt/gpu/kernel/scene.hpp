/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include "bvh.hpp"
#include "complex.hpp"

/*
 * Scene data for GPU rendering.
 *
 * The host compiles a wave_tracer scene into these plain structs
 * (src/gpu/renderer.cpp), and rejects anything the GPU does not implement
 * with an explanation. The GPU evaluates no spectra: the host draws
 * wavenumbers, as the CPU draws them for its paths, and evaluates everything
 * that depends on the wavenumber (emitters, BSDFs, the sensor's beams and
 * response) at each, into a `spectral_sample` and per-sample arrays of
 * emitters and materials. Each path takes one spectral sample. All lengths are
 * in metres, radiometric quantities in SI units.
 */

namespace wt::gpu::kernel {

constexpr u32 no_triangle = 0xffffffffu;
constexpr u32 no_edge = 0xffffffffu;
constexpr u32 no_emitter = 0xffffffffu;

/** @brief Shading data of a BVH triangle (`bvh_view::tris` order). */
struct tri_shading {
    vec3 n[3];      // decoded vertex normals (`mesh::triangle_t::n`)
    vec3 dpdu;      // `mesh::triangle_t::tangent_frame.dpdu`
    u32 material;
    u32 edges[3];   // `ads::tri_t::edge_ab, edge_bc, edge_ca`, or no_edge
    u32 emitter;    // the area emitter of the triangle's shape, or no_emitter
};

/**
 * @brief A geometric edge (`ads::edge_t`): of one triangle, or shared by two.
 *        Face normals point out of the wedge.
 */
struct edge {
    vec3 a, b;
    vec3 n1, t1;    // face normal, and tangent pointing into the face, of triangle 1
    vec3 n2, t2;    // ... of triangle 2
    real alpha;     // wedge opening angle (radians)
    u32 tri1, tri2; // BVH triangles; tri2 is no_triangle for an edge of one triangle
};

/** @brief UV coordinates of a BVH triangle's vertices (`mesh::triangle_t::uv`, or 0). */
struct tri_uv {
    vec2 uv[3];
};

constexpr u32 no_texture = 0xffffffffu;

constexpr u32 layout_L = 0;               // `bitmap::pixel_layout_e`
constexpr u32 layout_LA = 1;
constexpr u32 layout_RGB = 2;
constexpr u32 layout_RGBA = 3;

constexpr u32 filter_nearest = 0;         // `bitmap::texture_filter_type_e`
constexpr u32 filter_bilinear = 1;
constexpr u32 filter_bicubic = 2;

constexpr u32 wrap_black = 0;             // `bitmap::texture_wrap_mode_e`
constexpr u32 wrap_white = 1;
constexpr u32 wrap_clamp = 2;
constexpr u32 wrap_repeat = 3;
constexpr u32 wrap_mirror = 4;

constexpr u32 clamp_none = 0;             // `bitmap::texel_clamp_mode_e`
constexpr u32 clamp_non_negative = 1;
constexpr u32 clamp_non_positive = 2;

/**
 * @brief A bitmap (`bitmap::texture2d_t`): its texels, linear floating point
 *        in the bitmap's own layout (`components` per texel, row by row), are
 *        `plt_scene::texels[offset ..]`.
 */
struct bitmap {
    u32 offset;
    u32 width, height;
    u32 layout;
    u32 components;
    u32 filter;
    u32 uwrap, vwrap;
    u32 clamp;
    /**
     * @brief The value of texels outside the bitmap, with black or white
     *        wrapping, in u and in v (the CPU's `texel_native`).
     */
    real outside_u, outside_v;
};

/**
 * @brief A texture that varies over surfaces: a bitmap, looked up at affinely
 *        transformed UV coordinates (`texture::transform_t`). Its value at a
 *        wavenumber is a constant factor (of the constant textures that scale
 *        it, see `material`) times the bitmap's: the bitmap's luminance, or
 *        its RGB colour uplifted to the wavenumber (`bitmap_t::f`).
 */
struct texture {
    u32 bitmap;
    u32 rgb;                    // the bitmap is RGB or RGBA: its colour is uplifted
    real M[4];                  // row-major: looked up at M*uv + t
    real t[2];
};

/**
 * @brief What of a material does not depend on the wavenumber: its normal
 *        map (`bsdf::normalmap_t`), which perturbs the shading frame.
 */
struct material_shading {
    u32 normalmap;              // texture, or no_texture
    u32 normalmap_flip;         // flip the normal map's x and y
};

/** @brief No BSDF at the spectral sample's wavenumber: paths end here (`composite_t` without a matching bin). */
constexpr u32 material_absorbing = 0;
/**
 * @brief `surface_spm_t`: specular reflection and refraction, and, with a rough
 *        surface profile, scattering about them.
 */
constexpr u32 material_specular = 1;
/** @brief `diffuse_t`: Lambertian reflection. */
constexpr u32 material_diffuse = 2;

constexpr u32 profile_dirac = 0;          // `surface_profile::dirac_t`: smooth
constexpr u32 profile_fractal = 1;        // `surface_profile::fractal_t`
constexpr u32 profile_gaussian = 2;       // `surface_profile::gaussian_t`

/**
 * @brief A material at a spectral sample's wavenumber. Values with a texture
 *        (other than no_texture) are a constant factor, to be multiplied by
 *        the texture's value where a path hits (see `resolve_material`).
 */
struct material {
    u32 type;
    u32 two_sided;              // wrapped in `two_sided_t`
    u32 delta_only;             // `bsdf_t::is_delta_only`: no evaluation, nor next-event estimation
    real scale;                 // product of the `scale_t` wrappers' scales
    u32 scale_texture;
    // specular
    u32 has_transmission;       // `IOR_has_transmission(eta)` of src/bsdf/surface_spm.cpp
    complex eta;                // `surface_spm_t::IOR(k)`: exterior over interior refractive index
    real reflection_scale;      // `surface_spm_t::reflectivity_scale(k)`
    real transmission_scale;    // `surface_spm_t::transmissivity_scale(k)`
    /**
     * @brief The surface profile, with dimensionless parameters at the
     *        wavenumber k (the CPU's mm-based parameters combine with k so):
     *        the specular fraction is exp(-(|wi.z|+|wo.z|)² * `alpha_k`);
     *        fractal: `K` = k²T, `gamma`; gaussian: `s2` = σ²/k²; both:
     *        `sigma2_norm`, the PSD's normalization.
     */
    u32 profile;
    real alpha_k;
    real K;
    real gamma;
    real s2;
    real sigma2_norm;
    /**
     * @brief A textured roughness (of a roughness-parametrized profile): the
     *        parameters above derive from it where a path hits.
     */
    real roughness;
    u32 roughness_texture;
    // diffuse
    real reflectance;           // `diffuse_t`'s reflectance, clamped to [0,1] (once textured)
    u32 reflectance_texture;
};

constexpr u32 emitter_point = 0;          // `emitter::point_t`
constexpr u32 emitter_spot = 1;           // `emitter::spot_t`
constexpr u32 emitter_directional = 2;    // `emitter::directional_t`
constexpr u32 emitter_area = 3;           // `emitter::area_t`, of a constant radiance

/** @brief An emitter, but for what depends on the wavenumber (see `emission`). */
struct emitter {
    u32 type;
    real pmf;                   // probability of sampling this emitter (`scene_t::pdf_emitter`)

    // point, spot
    vec3 position;

    // spot: directions in its frame, whose z axis is its mean direction
    real dir_to_world[9];       // row-major: `to_world(dir3_t)`, before normalization
    real dir_to_local[9];       // row-major: `to_world.inverse()(dir3_t)`, before normalization
    real cos_cutoff, cos_falloff;
    real cutoff;                // radians
    real recp_cutoff_range;     // one over (cutoff - falloff), in 1/radians
    real cutoff_solid_angle;    // of the cone of directions sampled, in steradians

    // directional: radiates into a target disc, from outside the world
    frame target;               // n: the direction to the emitter
    vec3 target_centre;
    real target_radius;
    real far;                   // distance along n from the target to outside the world

    // area: triangles `area_tris[first_tri .. first_tri+tri_count-1]`
    u32 first_tri, tri_count;
    real area;                  // the shape's surface area
};

/**
 * @brief An emitter at a spectral sample's wavenumber, divided by the spectral
 *        sampling density (of `plt_path::integrate_forward` or `integrate_backward`).
 */
struct emission {
    // sourced beams: isotropic cones (`sourcing_geometry`)
    real tan_alpha;
    real x0;
    /**
     * @brief Forward transport: Watts per unit of a sourced beam's relative
     *        Stokes parameters (the sampled beam's flux, `emitter_t::sample`).
     */
    real flux;
    /**
     * @brief Backward transport: the radiant intensity (W/sr) of point and
     *        spot emitters, the irradiance (W/m²) of directional emitters, the
     *        radiance (W/m²/sr) of area emitters.
     */
    real intensity;
};

constexpr u32 sensor_virtual_plane = 0;
constexpr u32 sensor_perspective = 1;

/**
 * @brief A `sensor::virtual_plane_sensor_t` (`type` sensor_virtual_plane), or
 *        a `sensor::perspective_t` (sensor_perspective; backward transport only).
 *        The beams it sources depend on the wavenumber: see `spectral_sample`.
 */
struct sensor_desc {
    u32 type;

    // virtual plane
    vec3 origin;                // corner of element (0,0)
    frame f;                    // t,b span the plane; n is its front
    vec3 quad[4];               // the corners `Si` intersects: a, b, c, d
    vec2 extent;
    vec2 element_extent;
    vec2 recp_element_extent;
    real area;
    real recp_area;
    real importance;            // `virtual_plane_sensor_t::importance()`

    // perspective
    vec3 position;
    real to_camera[16];         // `sensor_to_camera_trns`, row-major: film position to camera-space point
    real dir_to_world[9];       // `sensor_transform`'s inverse transpose, row-major: directions to world
    vec3 ddir_dx, ddir_dy;      // `direction_pdvx/y()`
};

/** @brief Largest supported film reconstruction filter radius, in elements. */
constexpr i32 max_rf_radius = 8;

/**
 * @brief A `sensor::film_t` (2D, scalar). Elements are accumulated by the
 *        backend. The sensor's response depends on the wavenumber: see
 *        `spectral_sample`.
 */
struct film_desc {
    u32 width, height;
    u32 components;
    i32 rf_radius;              // Gaussian reconstruction filter radius, in elements
    real rf_sigma;              // ... and standard deviation
};

/** @brief The sensor's response of each film component (`response_t::f`) at a wavenumber. */
struct film_response {
    real c[4];
};

/**
 * @brief A wavenumber drawn for paths of an emitter (`scene_t::sample_spectrum`),
 *        and what depends on it, other than emitters and materials.
 */
struct spectral_sample {
    real k;                     // wavenumber (1/m)
    // beams the sensor sources (backward transport): isotropic cones (`sourcing_geometry`)
    real sensor_tan_alpha;
    real sensor_x0;
    film_response response;
    /**
     * @brief `RGB_to_spectral::uplift` at the wavenumber: the spectral value
     *        of the white, cyan, magenta, yellow, red, green and blue bases.
     */
    real uplift[7];
};

/** @brief `integrator::plt_path_t::options_t`, and how the scene is rendered. */
struct plt_path_params {
    u32 max_depth;
    u32 russian_roulette;
    u32 fsd;
    u32 ray_tracing_only;       // `sensor_t::ray_trace_only()`
    u32 backward;               // transport direction: paths start at the sensor
};

/**
 * @brief Everything a plt_path kernel reads.
 *
 * Spectral samples come in tables: a table holds `samples_per_emitter`
 * samples of each emitter, emitter `e`'s from `base + e*samples_per_emitter`
 * (`base` the table's first). A path samples an emitter, then takes one of
 * the emitter's samples (see `start_path`).
 */
struct plt_scene {
    bvh_view bvh;
    const WT_GPU_GLOBAL tri_shading* tris;
    const WT_GPU_GLOBAL edge* edges;

    const WT_GPU_GLOBAL spectral_sample* spectra;
    const WT_GPU_GLOBAL emission* emissions;        // emitter e at spectral sample s: [s*emitter_count + e]
    const WT_GPU_GLOBAL material* materials;        // material m at spectral sample s: [s*material_count + m]
    u32 emitter_count;
    u32 material_count;
    u32 samples_per_emitter;

    const WT_GPU_GLOBAL emitter* emitters;
    const WT_GPU_GLOBAL real* emitter_cdf;          // inclusive CDF of the emitter sampling distribution
    // area emitters' triangles (BVH triangles), and each emitter's inclusive CDF of them by area
    const WT_GPU_GLOBAL u32* area_tris;
    const WT_GPU_GLOBAL real* area_cdf;

    // textures: the triangles' UVs (null if no material has textures), materials' shading
    const WT_GPU_GLOBAL tri_uv* tri_uvs;
    const WT_GPU_GLOBAL material_shading* material_shadings;
    const WT_GPU_GLOBAL texture* textures;
    const WT_GPU_GLOBAL bitmap* bitmaps;
    const WT_GPU_GLOBAL float* texels;

    sensor_desc sensor;
    film_desc film;
    plt_path_params params;
};

}
