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

namespace wt { namespace gpu { namespace compute_kernel {

WT_GPU_CONSTANT constexpr u32 no_triangle = 0xffffffffu;
WT_GPU_CONSTANT constexpr u32 no_edge = 0xffffffffu;
WT_GPU_CONSTANT constexpr u32 no_emitter = 0xffffffffu;

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

WT_GPU_CONSTANT constexpr u32 no_texture = 0xffffffffu;

WT_GPU_CONSTANT constexpr u32 layout_L = 0;               // `bitmap::pixel_layout_e`
WT_GPU_CONSTANT constexpr u32 layout_LA = 1;
WT_GPU_CONSTANT constexpr u32 layout_RGB = 2;
WT_GPU_CONSTANT constexpr u32 layout_RGBA = 3;

WT_GPU_CONSTANT constexpr u32 filter_nearest = 0;         // `bitmap::texture_filter_type_e`
WT_GPU_CONSTANT constexpr u32 filter_bilinear = 1;
WT_GPU_CONSTANT constexpr u32 filter_bicubic = 2;

WT_GPU_CONSTANT constexpr u32 wrap_black = 0;             // `bitmap::texture_wrap_mode_e`
WT_GPU_CONSTANT constexpr u32 wrap_white = 1;
WT_GPU_CONSTANT constexpr u32 wrap_clamp = 2;
WT_GPU_CONSTANT constexpr u32 wrap_repeat = 3;
WT_GPU_CONSTANT constexpr u32 wrap_mirror = 4;

WT_GPU_CONSTANT constexpr u32 clamp_none = 0;             // `bitmap::texel_clamp_mode_e`
WT_GPU_CONSTANT constexpr u32 clamp_non_negative = 1;
WT_GPU_CONSTANT constexpr u32 clamp_non_positive = 2;

WT_GPU_CONSTANT constexpr u32 texel_float32 = 0;          // 32-bit floating point, linear (of EXR files)
WT_GPU_CONSTANT constexpr u32 texel_unorm8 = 1;           // 8-bit unsigned normalized integers (of 8-bit PNG files)
WT_GPU_CONSTANT constexpr u32 texel_unorm16 = 2;          // 16-bit ... (of 16-bit PNG files)

/**
 * @brief A bitmap (`bitmap::texture2d_t`): its texels' components, in the
 *        bitmap's own layout (`components` per texel, row by row), packed
 *        in 32-bit words from `plt_scene::texels[offset]` on, as the CPU
 *        stores them (`format`): floats (as their bits), or unsigned
 *        normalized integers (lowest bits first), which decode to linear
 *        values through tables (see `plt_scene::texel_luts`).
 */
struct bitmap {
    u32 offset;
    u32 format;
    /**
     * @brief unorm texels: the tables of their components' linear values
     *        (`colour_encoding_t::to_linear_fp`; alpha: `to_fp`), from
     *        `plt_scene::texel_luts[colour_lut]` and `[alpha_lut]` on.
     */
    u32 colour_lut, alpha_lut;
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
WT_GPU_CONSTANT constexpr u32 material_absorbing = 0;
/**
 * @brief `surface_spm_t`: specular reflection and refraction, and, with a rough
 *        surface profile, scattering about them.
 */
WT_GPU_CONSTANT constexpr u32 material_specular = 1;
/** @brief `diffuse_t`: Lambertian reflection. */
WT_GPU_CONSTANT constexpr u32 material_diffuse = 2;
/** @brief `dielectric_t`: Fresnel reflection and refraction at a smooth dielectric interface. */
WT_GPU_CONSTANT constexpr u32 material_dielectric = 3;

WT_GPU_CONSTANT constexpr u32 profile_dirac = 0;          // `surface_profile::dirac_t`: smooth
WT_GPU_CONSTANT constexpr u32 profile_fractal = 1;        // `surface_profile::fractal_t`
WT_GPU_CONSTANT constexpr u32 profile_gaussian = 2;       // `surface_profile::gaussian_t`

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
    // specular and dielectric (whose `eta` is real)
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
#if defined(REALISTIC_REMOTE_SENSING)
    /**
     * @brief A `remote_sensing::spectral_mix_t` reflectance (if nonzero): the texture's RGBA (unclamped and not
     *        uplifted) weights the mixed spectra's values, `reflectance_mix`, rather than its value.
     */
    u32 reflectance_mixed;
    real reflectance_mix[4];
#endif
};

WT_GPU_CONSTANT constexpr u32 emitter_point = 0;          // `emitter::point_t`
WT_GPU_CONSTANT constexpr u32 emitter_spot = 1;           // `emitter::spot_t`
WT_GPU_CONSTANT constexpr u32 emitter_directional = 2;    // `emitter::directional_t`
WT_GPU_CONSTANT constexpr u32 emitter_area = 3;           // `emitter::area_t`, of a constant radiance

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
 *        sampling density (of `plt_path::integrate_forward`, or of
 *        `integrate_backward` and plt_bdpt).
 */
struct emission {
    // sourced beams: isotropic cones (`sourcing_geometry`)
    real tan_alpha;
    real x0;
    /**
     * @brief Forward transport and plt_bdpt's emitter subpaths: Watts per unit
     *        of a sourced beam's relative Stokes parameters (the sampled beam's
     *        flux, `emitter_t::sample`).
     */
    real flux;
    /**
     * @brief Backward transport and plt_bdpt's sensor subpaths: the radiant
     *        intensity (W/sr) of point and spot emitters, the irradiance (W/m²)
     *        of directional emitters, the radiance (W/m²/sr) of area emitters.
     */
    real intensity;
};

WT_GPU_CONSTANT constexpr u32 sensor_virtual_plane = 0;
WT_GPU_CONSTANT constexpr u32 sensor_perspective = 1;
#if defined(REALISTIC_REMOTE_SENSING)
/** @brief `remote_sensing::telescope_t`: see `telescope_desc` (backward plt_path only). */
WT_GPU_CONSTANT constexpr u32 sensor_telescope = 2;
#endif

/**
 * @brief A `sensor::virtual_plane_sensor_t` (`type` sensor_virtual_plane), or
 *        a `sensor::perspective_t` (sensor_perspective; not for forward plt_path),
 *        or a telescope (sensor_telescope, described by `plt_scene::telescope`).
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
    real to_film[16];           // its inverse (`point_on_film`)
    real dir_to_world[9];       // `sensor_transform`'s inverse transpose, row-major: directions to world
    real dir_to_local[9];       // ... and of its inverse: directions to camera space
    vec3 ddir_dx, ddir_dy;      // `direction_pdvx/y()`
    /**
     * @brief `recp_solid_angle_density_for_sample(d)` over d.z³: the sensor's
     *        area over the image plane's distance squared.
     */
    real recp_sa_scale;
    real near_clip;             // `near_clip()`: rays start this far along
};

/** @brief Largest supported film reconstruction filter radius, in elements. */
WT_GPU_CONSTANT constexpr i32 max_rf_radius = 8;

/**
 * @brief A `sensor::film_t` (2D, scalar or polarimetric). Elements are accumulated by the
 *        backend. The sensor's response depends on the wavenumber: see
 *        `spectral_sample`.
 */
struct film_desc {
    u32 width, height;
    u32 components;
    /**
     * @brief Elements' components are Stokes parameters vectors (4 values)
     *        rather than scalars: paths run as `path_state_t<true>`.
     */
    u32 polarimetric;
    i32 rf_radius;              // Gaussian reconstruction filter radius, in elements
    real rf_sigma;              // ... and standard deviation
};

/** @brief The sensor's response of each film component (`response_t::f`) at a wavenumber. */
struct film_response {
    real c[4];
};

#if defined(REALISTIC_REMOTE_SENSING)
/*
 * The telescope sensor (`remote_sensing::telescope_t`, of src/remote_sensing/telescope.cpp, and its optics,
 * `remote_sensing::optics::telescope_optics_t`): its elements, focal plane and placement, as the host computed them
 * when loading, in the telescope's frame (metres). The coatings' refractive indices depend on the wavenumber: see
 * `spectral_sample::telescope_iors`. See telescope.hpp.
 */

WT_GPU_CONSTANT constexpr u32 telescope_max_elements = 32;
WT_GPU_CONSTANT constexpr u32 telescope_max_aspheric = 8;      // r⁴ to r¹⁸
WT_GPU_CONSTANT constexpr u32 telescope_max_layers = 16;        // of all coatings
WT_GPU_CONSTANT constexpr u32 telescope_max_iors = 16;          // of all coatings, substrates and layers
WT_GPU_CONSTANT constexpr u32 telescope_aim_grid_size = 5;      // `telescope_t::aim_grid_size`

WT_GPU_CONSTANT constexpr u32 telescope_mirror = 0;             // `optics::element_kind_e`
WT_GPU_CONSTANT constexpr u32 telescope_stop = 1;
WT_GPU_CONSTANT constexpr u32 telescope_obscuration = 2;
WT_GPU_CONSTANT constexpr u32 telescope_tube = 3;

/** @brief A rough mirror's TIS table (`optics::abg_scatter_t::tis_samples`): at |β₀| = 0, 1/64, ..., 1. */
WT_GPU_CONSTANT constexpr u32 telescope_tis_samples = 65;
/** @brief The bins of a rough mirror's sampling table (`optics::abg_scatter_t::sampling_bins`). */
WT_GPU_CONSTANT constexpr u32 telescope_scatter_bins = 256;
WT_GPU_CONSTANT constexpr u32 telescope_max_scatters = 8;      // distinct mirror scatter models
/** @brief A smooth mirror (`telescope_element::scatter`). */
WT_GPU_CONSTANT constexpr u32 telescope_smooth = 0xffffffffu;
/** @brief Most reflections a ray traced non-sequentially makes (`telescope_optics_t::max_events`). */
WT_GPU_CONSTANT constexpr u32 telescope_max_events = 32;

/** @brief An ideal coating's substrate (`telescope_element::substrate`). */
WT_GPU_CONSTANT constexpr u32 telescope_ideal = 0xffffffffu;

/** @brief `optics::element_t`. */
struct telescope_element {
    u32 kind;
    // its surface (`optics::surface_shape_t`), and clear aperture (`optics::aperture_t`)
    real curvature, conic;
    u32 aspheric_count;
    real aspheric[telescope_max_aspheric];
    vec2 aperture_centre;
    real aperture_radius2, hole_radius2;    // squared radii
    real length;                            // a tube's (along local z)
    /** @brief A rough mirror's scatter: `telescope_desc::scatters[scatter]`, or telescope_smooth. */
    u32 scatter;
    /** @brief Stops, obscurations and tubes: their Lambertian reflectance (stray light). */
    real reflectance;
    /** @brief A mirror's reflective side (`element_t::reflective_side`): +1 its local +z side, -1 the other, 0 both. */
    i32 reflective_side;
    real to_telescope[12];                  // rigid, row-major 3x4: element frame to telescope frame
    real to_local[12];                      // ... and back
    /**
     * @brief A mirror's coating (`optics::coating_t`): ideal (`substrate` telescope_ideal), or a substrate of
     *        refractive index `spectral_sample::telescope_iors[substrate]` under `telescope_desc::layers`
     *        `first_layer` to `first_layer+layer_count-1`, the outermost first.
     */
    u32 substrate;
    u32 first_layer, layer_count;
};

/** @brief A thin film of a coating (`optics::coating_t::layer_t`). */
struct telescope_layer {
    u32 ior;                                // `spectral_sample::telescope_iors[ior]`
    real thickness;                         // metres
};

/**
 * @brief `optics::abg_scatter_t`: a rough mirror's ABg scatter, its TIS table, and its sampling table (the bins'
 *        radii, and the inclusive CDF of their masses).
 */
struct telescope_scatter {
    real A, B, g;
    real tis[telescope_tis_samples];
    real edges[telescope_scatter_bins+1];
    real cdf[telescope_scatter_bins+1];
};

/** @brief `optics::cone_t`, by its half-angle (radians). */
struct telescope_cone {
    vec3 axis;
    real half_angle;
};

/** @brief A `remote_sensing::telescope_t`. */
struct telescope_desc {
    telescope_element elements[telescope_max_elements];
    u32 element_count;
    /** @brief The elements rays pass in turn (mirrors and stops), in the order light meets them. */
    u32 sequence[telescope_max_elements];
    u32 sequence_count;
    telescope_layer layers[telescope_max_layers];
    telescope_scatter scatters[telescope_max_scatters];

    /** @brief The sphere bounding the clear aperture nearest the focal plane (`telescope_optics_t::aim`). */
    vec3 aim_centre;
    real aim_radius;
    /** @brief The cones of passing directions, row-major on a grid spanning the film (`telescope_t::aim_grid`). */
    telescope_cone aim_grid[telescope_aim_grid_size*telescope_aim_grid_size];
    real aim_margin;                        // `telescope_t::aim_margin`
    real aim_tight_fraction;                // `telescope_t::aim_tight_fraction`

    // the focal plane (`telescope_t::focal_plane_t`)
    real fp_to_telescope[12];               // rigid, row-major 3x4
    real fp_curvature;                      // 1/metres; 0 if flat
    real pixel_pitch;                       // metres

    // its placement (`telescope_t::placement_t`)
    real rotation[9];                       // row-major: telescope frame to world
    real near_clip;                         // metres
    /**
     * @brief The film centre's line of sight (`telescope_t::properties_t`), in the telescope's frame and in the
     *        world; and the world point `near_clip` along it from the telescope's origin, which rays that start
     *        near the scene start about (in the kernels' precision, the telescope's position is too far).
     */
    vec3 line_of_sight;
    vec3 world_line_of_sight;
    vec3 near_origin;
    /** @brief One over the film centre's projected solid angle (sr): samples' importance is relative to it. */
    real recp_projected_solid_angle;

    /**
     * @brief Stray light (`telescope_t::stray_light_t`): traced non-sequentially if nonzero; a rough mirror scatters
     *        with probability at least `scatter_probability`; `hemisphere_fraction` of the directions are drawn from
     *        the hemisphere on the side of `front` (the focal plane's normal at its centre) about the focal surface's
     *        normal; the focal-plane array (its surface, of curvature `fp_curvature`, in the frame `fp_to_local` maps
     *        to, within `fp_half_extent` in x and y) absorbs.
     */
    u32 stray_light;
    real scatter_probability;
    real hemisphere_fraction;
    vec3 front;
    real fp_to_local[12];
    vec2 fp_half_extent;
};
#endif

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
    /**
     * @brief plt_bdpt without MIS: the spectral weight of a contribution
     *        (`1/((s+t+1)*k_density)`, without the 1/(s+t+1)), over the one
     *        the one
     *        the emissions include (see `emission`).
     */
    real no_mis_scale;
#if defined(REALISTIC_REMOTE_SENSING)
    /** @brief A telescope sensor's coatings' refractive indices (n + ik) at the wavenumber (see `telescope_element`). */
    complex telescope_iors[telescope_max_iors];
#endif
};

/**
 * @brief The integrator, `integrator::plt_path_t` or `integrator::plt_bdpt_t`,
 *        its options, and how the scene is rendered.
 */
struct plt_params {
    u32 bdpt;                   // plt_bdpt (else plt_path)
    u32 max_depth;
    u32 russian_roulette;
    u32 fsd;                    // free-space diffraction (of beams; plt_bdpt's: Fraunhofer, fraunhofer.hpp)
    u32 ray_tracing_only;       // `sensor_t::ray_trace_only()`
    u32 backward;               // plt_path's transport direction: paths start at the sensor
    u32 mis;                    // plt_bdpt: multiple importance sampling of the connection strategies
    u32 sensor_direct;          // plt_bdpt: connects emitter subpaths to sampled sensor points
    u32 emitter_direct;         // plt_bdpt: connects sensor subpaths to sampled emitter points

    /** @brief Paths start at the sensor: the spectral weighting and sensor beams of backward transport. */
    WT_GPU_FN bool sensor_paths() const { return backward!=0 || bdpt!=0; }
    /** @brief Splats into sensor elements' blocks: the film's image, of values and reconstruction weights. */
    WT_GPU_FN bool block_splats() const { return backward!=0 || bdpt!=0; }
};

/**
 * @brief Everything a plt_path or plt_bdpt kernel reads.
 *
 * Spectral samples come in tables: a table holds `samples_per_emitter`
 * samples of each sampled emitter, emitter `e`'s from
 * `base + e*samples_per_emitter` (`base` the table's first). A path samples
 * an emitter, then takes one of the emitter's samples (see `start_path`).
 * The first `sampled_emitter_count` emitters are sampled, by `emitter_cdf`;
 * the rest are area emitters of no sampling probability (e.g. rounded to
 * zero beside far brighter ones: stars by the sun), which paths only hit.
 */
struct plt_scene {
    bvh_view bvh;
    const WT_GPU_GLOBAL tri_shading* tris;
    const WT_GPU_GLOBAL edge* edges;

    const WT_GPU_GLOBAL spectral_sample* spectra;
    const WT_GPU_GLOBAL emission* emissions;        // emitter e at spectral sample s: [s*emitter_count + e]
    const WT_GPU_GLOBAL material* materials;        // material m at spectral sample s: [s*material_count + m]
    u32 emitter_count;
    u32 sampled_emitter_count;
    u32 material_count;
    u32 samples_per_emitter;

    const WT_GPU_GLOBAL emitter* emitters;
    const WT_GPU_GLOBAL real* emitter_cdf;          // inclusive CDF of the emitter sampling distribution (sampled emitters')
    // area emitters' triangles (BVH triangles), and each emitter's inclusive CDF of them by area
    const WT_GPU_GLOBAL u32* area_tris;
    const WT_GPU_GLOBAL real* area_cdf;

    // textures: the triangles' UVs (null if no material has textures), materials' shading
    const WT_GPU_GLOBAL tri_uv* tri_uvs;
    const WT_GPU_GLOBAL material_shading* material_shadings;
    const WT_GPU_GLOBAL texture* textures;
    const WT_GPU_GLOBAL bitmap* bitmaps;
    const WT_GPU_GLOBAL u32* texels;
    const WT_GPU_GLOBAL real* texel_luts;           // decoding tables of unorm texels (see `bitmap`)

    // plt_bdpt's free-space diffraction: the sampler's lookup tables (see fraunhofer.hpp), or null
    const WT_GPU_GLOBAL real* fsd_lut;

#if defined(REALISTIC_REMOTE_SENSING)
    // a telescope sensor (sensor_telescope), or null
    const WT_GPU_GLOBAL telescope_desc* telescope;
#endif

    sensor_desc sensor;
    film_desc film;
    plt_params params;
};

} } }
