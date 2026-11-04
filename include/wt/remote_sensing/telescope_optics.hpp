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
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include <wt/math/common.hpp>
#include <wt/remote_sensing/telescope_coating.hpp>

/**
 * @brief A telescope's optics, traced as rays in double precision in the telescope's own frame (metres): the
 *        optical surfaces of a `telescope` sensor (see `wt::remote_sensing::telescope_t`).
 *
 *        The telescope frame follows optical design software's convention: light from the scene arrives
 *        travelling towards +z. Each element has its own frame (its vertex at the origin, its axis local +z),
 *        placed in the telescope frame by a rigid transform; a surface's radius of curvature is positive when its
 *        centre of curvature is at local +z (as in Zemax, CODE V or Optiland, for surfaces not flipped).
 *
 *        The elements are the telescope's physical objects, listed in the order light meets them. Mirrors and
 *        stops are sequential: a ray passes each, in order, within its clear aperture. Every mirror and
 *        obscuration is also an obstacle to all the ray's other segments (the secondary mirror shadows the primary,
 *        the primary blocks rays outside its central hole). The same elements are meant to serve non-sequential
 *        (stray light) tracing later.
 */
namespace wt::remote_sensing::optics {

/**
 * @brief A ray in double precision: origin and (unit) direction.
 */
struct ray_t {
    glm::dvec3 o, d;
};

/**
 * @brief A rotationally symmetric surface, in its own frame: the even asphere
 *        z(r) = c r² / (1 + √(1 - (1+κ) c² r²)) + a₄ r⁴ + a₆ r⁶ + ...
 *        with c the vertex curvature (1/radius; 0 for a plane), κ the conic constant (0 sphere, -1 paraboloid,
 *        < -1 hyperboloid, > -1 ellipsoid), and r² = x² + y².
 */
struct surface_shape_t {
    /** @brief Vertex curvature (1/m). */
    double curvature = 0;
    /** @brief Conic constant. */
    double conic = 0;
    /** @brief Coefficients of r⁴, r⁶, ..., in metres (a₂ᵢ in m^(1-2i)). */
    std::vector<double> aspheric;

    /**
     * @brief The sag z at r² = x² + y², or nothing beyond the conic's extent.
     */
    [[nodiscard]] std::optional<double> sag(double r2) const noexcept;
    /**
     * @brief The unit normal at (x,y) on the surface, facing local +z, or nothing beyond the conic's extent.
     */
    [[nodiscard]] std::optional<glm::dvec3> normal(double x, double y) const noexcept;
    /**
     * @brief The distance along the (local) ray `r` to the surface, beyond `tmin`: on the sheet through the vertex
     *        (for a sphere, the cap about the vertex, not the far side). Nothing if the ray misses it.
     */
    [[nodiscard]] std::optional<double> intersect(const ray_t& r, double tmin) const noexcept;
};

/**
 * @brief A clear aperture on a surface: within `radius` of `centre` (in the surface's local x,y; off the vertex
 *        for an off-axis segment of a parent surface), and outside `hole_radius` of it.
 */
struct aperture_t {
    double radius = 0;
    double hole_radius = 0;
    glm::dvec2 centre = { 0,0 };

    [[nodiscard]] inline bool contains(double x, double y) const noexcept {
        const auto r2 = (x-centre.x)*(x-centre.x) + (y-centre.y)*(y-centre.y);
        return r2<=radius*radius && r2>=hole_radius*hole_radius;
    }
};

enum class element_kind_e : std::uint8_t {
    /** @brief Reflects, within its aperture (sequential). */
    mirror,
    /** @brief An opening in a plane: passes rays within its aperture (sequential), e.g. an aperture or field stop. */
    stop,
    /** @brief Blocks rays within its aperture, wherever they cross it (not sequential): e.g. a baffle. */
    obscuration,
};

/**
 * @brief An element of a telescope's optics.
 */
struct element_t {
    std::string id;
    element_kind_e kind = element_kind_e::mirror;
    /** @brief Stops and obscurations are flat. */
    surface_shape_t shape;
    aperture_t aperture;
    /** @brief A mirror's coating (`coating_t`; ideal if none). */
    std::shared_ptr<const coating_t> coating;
    /** @brief Element frame to telescope frame (rigid), and its inverse. */
    glm::dmat4 to_telescope = glm::dmat4(1), to_local = glm::dmat4(1);

    struct hit_t {
        double t;
        glm::dvec3 local;
    };
    /**
     * @brief Where the (telescope frame) ray `r` meets the element's surface beyond `tmin`: the distance and the
     *        point in the element's frame. Ignores the aperture.
     */
    [[nodiscard]] std::optional<hit_t> intersect(const ray_t& r, double tmin) const noexcept;
};

/**
 * @brief A cone of directions: unit axis and cosine of its half-angle.
 */
struct cone_t {
    glm::dvec3 axis;
    double cos_half_angle;

    [[nodiscard]] inline double solid_angle() const noexcept { return 2*std::numbers::pi*(1-cos_half_angle); }
    [[nodiscard]] inline bool contains(const glm::dvec3& d) const noexcept { return glm::dot(d, axis)>=cos_half_angle; }
    /**
     * @brief A direction of the cone, uniformly distributed, from two uniform numbers in [0,1).
     */
    [[nodiscard]] glm::dvec3 sample(double u1, double u2) const noexcept;
};

/**
 * @brief How the telescope sees the scene from a point on its focal plane: the rays from there that pass the optics.
 */
struct field_point_t {
    /** @brief Fraction of the aiming cone's rays that pass. */
    double passed_fraction = 0;
    /** @brief Projected solid angle (on the focal plane) of the rays that pass, sr: the point's throughput. */
    double projected_solid_angle = 0;
    /** @brief Mean direction of the rays leaving towards the scene (telescope frame). */
    glm::dvec3 line_of_sight = { 0,0,-1 };
    /** @brief RMS angle of those rays about the line of sight, radians: the geometric blur's size. */
    double rms_spread = 0;
    /** @brief Outer diameter of the bundle of rays leaving towards the scene (the entrance pupil), metres. */
    double pupil_diameter = 0;
    /** @brief Ratio of the bundle's inner (obscured) diameter to its outer. */
    double pupil_obscuration = 0;
    /** @brief The narrowest cone, about their mean, of the directions from the point that pass (of the samples). */
    cone_t passing_cone = { .axis = { 0,0,1 }, .cos_half_angle = 1 };
};

/**
 * @brief A telescope's optics.
 */
class telescope_optics_t {
public:
    static constexpr auto npos = std::numeric_limits<std::size_t>::max();
    /** @brief Most mirrors a telescope may have. */
    static constexpr std::size_t max_mirrors = 16;

    /**
     * @brief A reflection off a mirror, of a ray traced from the focal plane (telescope frame).
     */
    struct reflection_t {
        std::size_t element;
        /** @brief The mirror's unit normal there, facing the ray. */
        glm::dvec3 normal;
        /** @brief The ray's direction before and after the reflection. */
        glm::dvec3 d_in, d_out;
    };

    struct trace_t {
        bool passed = false;
        /** @brief Passed: the ray leaving the telescope towards the scene, from its last surface (telescope frame). */
        ray_t exit;
        /** @brief Not passed: the element that stopped the ray (missed, outside its aperture, or in its way). */
        std::size_t stopped_by = npos;
        /** @brief The ray's reflections, in the order traced (from the focal plane). */
        std::array<reflection_t, max_mirrors> reflections;
        std::uint32_t reflection_count = 0;
    };

private:
    std::vector<element_t> elems;
    std::vector<std::size_t> seq;
    // a sphere bounding the clear aperture of the sequential element nearest the focal plane
    glm::dvec3 aim_centre;
    double aim_radius;

    [[nodiscard]] std::size_t obstacle(const ray_t& r, double tmax,
                                       std::size_t from, std::size_t to) const noexcept;

public:
    /**
     * @brief Throws `std::invalid_argument` unless there are 1 to `max_mirrors` mirrors and every aperture is valid.
     */
    explicit telescope_optics_t(std::vector<element_t> elements);

    [[nodiscard]] inline const auto& elements() const noexcept { return elems; }
    /** @brief The elements a ray passes in turn (mirrors and stops), in the order light meets them. */
    [[nodiscard]] inline const auto& sequence() const noexcept { return seq; }
    /** @brief The sphere bounding the clear aperture of the sequential element nearest the focal plane (see `aim`). */
    [[nodiscard]] inline const auto& aim_sphere_centre() const noexcept { return aim_centre; }
    [[nodiscard]] inline double aim_sphere_radius() const noexcept { return aim_radius; }

    /**
     * @brief Traces a ray that leaves the focal plane backwards through the optics: through the sequential
     *        elements in reverse order, reflecting off the mirrors, and out towards the scene.
     */
    [[nodiscard]] trace_t trace_from_focal_plane(const ray_t& r) const noexcept;

    /**
     * @brief The cone, from focal-plane point `p`, of the rays that can reach the clear aperture of the sequential
     *        element nearest the focal plane: rays outside it cannot pass the optics.
     */
    [[nodiscard]] cone_t aim(const glm::dvec3& p) const noexcept;

    /**
     * @brief Traces `samples` rays (a Fibonacci lattice over the aiming cone) from focal-plane point `p`, whose
     *        focal plane's normal is `normal`.
     */
    [[nodiscard]] field_point_t analyse(const glm::dvec3& p, const glm::dvec3& normal,
                                        std::uint32_t samples) const noexcept;
};

}
