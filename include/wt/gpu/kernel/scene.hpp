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
 * implement is rejected there with an explanation. All lengths are in metres,
 * radiometric quantities in SI units.
 */

namespace wt::gpu::kernel {

constexpr u32 no_triangle = 0xffffffffu;
constexpr u32 no_edge = 0xffffffffu;

/** @brief Shading data of a BVH triangle (`bvh_view::tris` order). */
struct tri_shading {
    vec3 n[3];      // decoded vertex normals (`mesh::triangle_t::n`)
    vec3 dpdu;      // `mesh::triangle_t::tangent_frame.dpdu`
    u32 material;
    u32 edges[3];   // `ads::tri_t::edge_ab, edge_bc, edge_ca`, or no_edge
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

/** @brief No BSDF at the rendered wavenumber: paths end here (`composite_t` without a matching bin). */
constexpr u32 material_absorbing = 0;
/** @brief `surface_spm_t` with a Dirac surface profile: specular reflection and refraction. */
constexpr u32 material_specular = 1;
/** @brief `diffuse_t` with a constant reflectance: Lambertian reflection. */
constexpr u32 material_diffuse = 2;

struct material {
    u32 type;
    u32 two_sided;              // wrapped in `two_sided_t`
    real scale;                 // product of the `scale_t` wrappers' (constant) scales
    // specular
    u32 has_transmission;       // `IOR_has_transmission(eta)` of src/bsdf/surface_spm.cpp
    complex eta;                // `surface_spm_t::IOR(k)`: exterior over interior refractive index
    real reflection_scale;      // `surface_spm_t::reflectivity_scale(k)`
    real transmission_scale;    // `surface_spm_t::transmissivity_scale(k)`
    // diffuse
    real reflectance;           // `diffuse_t`'s reflectance, clamped to [0,1]
};

/** @brief An `emitter::point_t`. */
struct point_emitter {
    vec3 position;
    // sourced beams: isotropic cones (`point_t::sourcing_geometry`)
    real tan_alpha;
    real x0;
    /**
     * @brief Forward transport: Watts per unit of a beam's relative Stokes
     *        parameters: the sampled beam's flux (`point_t::sample`) divided by
     *        the spectral sampling density (`scene_t::sum_spectral_pdf_for_all_emitters`).
     */
    real flux;
    /**
     * @brief Backward transport: the radiant intensity (W/sr), the spectral
     *        radiant intensity divided by the spectral sampling density of
     *        `plt_path::integrate_backward`.
     */
    real intensity;
    real pmf;           // probability of sampling this emitter (`scene_t::pdf_emitter`)
};

constexpr u32 sensor_virtual_plane = 0;
constexpr u32 sensor_perspective = 1;

/**
 * @brief A `sensor::virtual_plane_sensor_t` (`type` sensor_virtual_plane), or
 *        a `sensor::perspective_t` (sensor_perspective; backward transport only).
 */
struct sensor_desc {
    u32 type;
    // beams the sensor sources (backward transport): isotropic cones (`sourcing_geometry`)
    real tan_alpha;
    real x0;

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
    u32 backward;               // transport direction: paths start at the sensor
    real k;                     // rendered wavenumber (1/m)
};

/** @brief Everything a plt_path kernel reads. */
struct plt_scene {
    bvh_view bvh;
    const WT_GPU_GLOBAL tri_shading* tris;
    const WT_GPU_GLOBAL edge* edges;
    const WT_GPU_GLOBAL material* materials;

    const WT_GPU_GLOBAL point_emitter* emitters;
    const WT_GPU_GLOBAL real* emitter_cdf;     // inclusive CDF of the emitter sampling distribution
    u32 emitter_count;

    sensor_desc sensor;
    film_desc film;
    plt_path_params params;
};

}
