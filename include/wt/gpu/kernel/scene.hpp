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
 * The host compiles a wave_tracer scene into these plain structs for one
 * rendered wavenumber (src/gpu/renderer.cpp): spectra, BSDFs, emitters and the
 * sensor are evaluated at that wavenumber, and anything the GPU does not
 * implement is rejected there with an explanation. All lengths are in metres.
 */

namespace wt::gpu::kernel {

/** @brief Shading data of a BVH triangle (`bvh_view::tris` order). */
struct tri_shading {
    vec3 n[3];      // decoded vertex normals (`mesh::triangle_t::n`)
    vec3 dpdu;      // `mesh::triangle_t::tangent_frame.dpdu`
    u32 material;
};

/** @brief No BSDF at the rendered wavenumber: paths end here (`composite_t` without a matching bin). */
constexpr u32 material_absorbing = 0;
/** @brief `surface_spm_t` with a Dirac surface profile: specular reflection and refraction. */
constexpr u32 material_specular = 1;

struct material {
    u32 type;
    u32 two_sided;              // wrapped in `two_sided_t`
    u32 has_transmission;       // `IOR_has_transmission(eta)` of src/bsdf/surface_spm.cpp
    complex eta;                // `surface_spm_t::IOR(k)`: exterior over interior refractive index
    real reflection_scale;      // `surface_spm_t::reflectivity_scale(k)`
    real transmission_scale;    // `surface_spm_t::transmissivity_scale(k)`
};

/** @brief An `emitter::point_t`. */
struct point_emitter {
    vec3 position;
    // sourced beams: isotropic cones (`point_t::sourcing_geometry`)
    real tan_alpha;
    real x0;
    /**
     * @brief Watts per unit of a beam's relative Stokes parameters: the
     *        sampled beam's flux (`point_t::sample`) divided by the spectral
     *        sampling density (`scene_t::sum_spectral_pdf_for_all_emitters`).
     */
    real flux;
};

/** @brief A `sensor::virtual_plane_sensor_t`. */
struct virtual_plane_sensor {
    vec3 origin;                // corner of element (0,0)
    frame f;                    // t,b span the plane; n is its front
    vec3 quad[4];               // the corners `Si` intersects: a, b, c, d
    vec2 recp_element_extent;
    real importance;            // `virtual_plane_sensor_t::importance()`
};

/** @brief Largest supported film reconstruction filter radius, in elements. */
constexpr i32 max_rf_radius = 8;

/** @brief A `sensor::film_t` (2D, scalar). Elements are accumulated by the backend. */
struct film_desc {
    u32 width, height;
    u32 components;
    i32 rf_radius;              // Gaussian reconstruction filter radius, in elements
    real rf_sigma;              // ... and standard deviation
    real response[4];           // sensor response of each component at the rendered wavenumber
};

/** @brief `integrator::plt_path_t::options_t`, and how the scene is rendered. */
struct plt_path_params {
    u32 max_depth;
    u32 russian_roulette;
    u32 fsd;
    u32 ray_tracing_only;       // `sensor_t::ray_trace_only()`
    real k;                     // rendered wavenumber (1/m)
};

/** @brief Everything a forward plt_path kernel reads. */
struct plt_scene {
    bvh_view bvh;
    const WT_GPU_GLOBAL tri_shading* tris;
    const WT_GPU_GLOBAL material* materials;

    const WT_GPU_GLOBAL point_emitter* emitters;
    const WT_GPU_GLOBAL real* emitter_cdf;     // inclusive CDF of the emitter sampling distribution
    u32 emitter_count;

    virtual_plane_sensor sensor;
    film_desc film;
    plt_path_params params;
};

}
