/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include <wt/sensor/sensor/film_backed_sensor.hpp>
#include <wt/sensor/film/film.hpp>
#include <wt/sensor/sensor_sample.hpp>
#include <wt/remote_sensing/telescope_optics.hpp>

#include <wt/beam/beam.hpp>
#include <wt/sampler/sampler.hpp>
#include <wt/math/common.hpp>

#include <wt/wt_context.hpp>

namespace wt::remote_sensing {

/**
 * @brief A satellite's camera that images the scene through a telescope's mirrors: a focal-plane array behind
 *        optics described by their surfaces (`optics::telescope_optics_t`), placed as a `satellite_frame` camera
 *        (`satellite_placement_t`).
 *
 *        Each sample takes a point in its pixel on the focal plane and a direction towards the optics, and traces
 *        the ray backwards through them, in double precision: rays that miss a clear aperture, or meet an obstacle
 *        (the secondary mirror's shadow, the primary's central hole) are vignetted. A ray that leaves the
 *        telescope starts a beam from the last surface it reflected off, towards the scene: of the pixel's angular
 *        extent, and of the narrowest cross section that extent allows (a minimum-uncertainty beam).
 *
 *        Directions are sampled from a mixture: mostly (`aim_tight_fraction`) from the cone of the directions that
 *        pass, found when loading on a grid of focal-plane points and interpolated, else from the cone that bounds
 *        the clear aperture nearest the focal plane (`optics::telescope_optics_t::aim`), which every passing
 *        direction is in. The mixture's density weighs the samples: without bias, should a tight cone miss some
 *        passing directions, and without wasting samples on an aperture larger than its beam (a tilted fold
 *        mirror's).
 *
 *        Each reflection applies its mirror's coating (`optics::coating_t`) as a Mueller matrix of its Fresnel
 *        coefficients, at the sample's wavelength and the ray's angle of incidence, composed in s-p frames as
 *        wave_tracer's surfaces compose theirs: from the detector's frame (`detector_frame`: polarimetric films
 *        record Stokes parameters about the film's axes) to the frame of the light arriving from the scene.
 *
 *        Pixels record radiance relative to the film centre's throughput: the samples' importance is the projected
 *        solid angle of the rays that pass, over the film centre's, and the coatings' Mueller matrices. A uniform scene renders as its radiance at
 *        the film's centre, and dimmer where the optics vignette; as a `satellite_frame` camera's, the render is the
 *        radiance at the entrance pupil, for a detector model to scale by the aperture.
 *
 *        Only backward transport is supported (backward `plt_path`): the sensor cannot connect to scene points
 *        (`sample_direct`), and has no GPU implementation yet.
 */
template <bool polarimetric>
class telescope_t final : public sensor::film_backed_sensor_t<2, polarimetric> {
public:
    // the film's reconstruction filter's scale, w.r.t. sensor element size (as a perspective sensor's)
    static constexpr f_t beam_source_spatial_stddev = .25;

    using film_t = sensor::film_backed_sensor_t<2,polarimetric>::film_t;

    /**
     * @brief The focal-plane array: pixel (i,j) spans [i,i+1)×[j,j+1) pixel pitches about the film's centre, on
     *        the plane z=0 of the focal plane's frame, at local x and y.
     */
    struct focal_plane_t {
        glm::dmat4 to_telescope;
        double pixel_pitch;
    };
    /**
     * @brief Telescope frame to world: rotation, and the telescope's origin (metres), and where rays start.
     */
    struct placement_t {
        glm::dmat3 rotation;
        glm::dvec3 position;
        double near_clip;
    };
    /**
     * @brief What the optics do, computed when loading.
     */
    struct properties_t {
        /** @brief Focal lengths along the film's x and y, metres: from the lines of sight about the film's centre. */
        double focal_length_x, focal_length_y;
        /** @brief The film centre's rays (see `optics::field_point_t`). */
        optics::field_point_t centre;
        /** @brief Throughput of the film's corners, relative to its centre's. */
        double corner_relative_throughput;
    };

    /** @brief The cones of passing directions are found on aim_grid_size² points spanning the film. */
    static constexpr std::uint32_t aim_grid_size = 5;
    /** @brief Interpolated, they are widened by this fraction of their half-angle. */
    static constexpr double aim_margin = .02;
    /** @brief The fraction of samples aimed within them. */
    static constexpr double aim_tight_fraction = .9;

private:
    std::shared_ptr<const optics::telescope_optics_t> optics;
    focal_plane_t fp;
    placement_t place;
    properties_t props;
    std::vector<optics::cone_t> aim_grid;      // row major, aim_grid_size²
    f_t tan_alpha;
    f_t phase_space_extent_scale;

    /**
     * @brief The cone of passing directions at a film position: the grid's, interpolated (the axis), and the
     *        widest of the four about it (the half-angle), widened by `aim_margin`.
     */
    [[nodiscard]] optics::cone_t passing_cone(const vec2_t& film_pos) const noexcept {
        constexpr auto n = aim_grid_size;
        const auto g = glm::clamp(glm::dvec2{ film_pos } / glm::dvec2{ this->film().dimensions() } * double(n-1),
                                  0., double(n-1));
        const auto i = glm::min(glm::uvec2{ g }, glm::uvec2{ n-2 });
        const auto f = g - glm::dvec2{ i };
        const auto& c00 = aim_grid[i.y*n + i.x];
        const auto& c10 = aim_grid[i.y*n + i.x+1];
        const auto& c01 = aim_grid[(i.y+1)*n + i.x];
        const auto& c11 = aim_grid[(i.y+1)*n + i.x+1];
        const auto axis = glm::normalize((1-f.x)*(1-f.y)*c00.axis + f.x*(1-f.y)*c10.axis +
                                         (1-f.x)*f.y*c01.axis + f.x*f.y*c11.axis);
        const auto cos_widest = std::min({ c00.cos_half_angle, c10.cos_half_angle,
                                           c01.cos_half_angle, c11.cos_half_angle });
        const auto half_angle = std::min(std::numbers::pi, std::acos(cos_widest) * (1+aim_margin));
        return { .axis = axis, .cos_half_angle = std::cos(half_angle) };
    }

    [[nodiscard]] inline glm::dvec3 focal_plane_point(const vec2_t& film_pos) const noexcept {
        const auto dims = glm::dvec2{ this->film().dimensions() };
        const auto local = (glm::dvec2{ film_pos } - dims/2.) * fp.pixel_pitch;
        return glm::dvec3{ fp.to_telescope * glm::dvec4{ local, 0, 1 } };
    }
    [[nodiscard]] inline glm::dvec3 focal_plane_normal() const noexcept {
        return glm::normalize(glm::dvec3{ fp.to_telescope * glm::dvec4{ 0,0,1,0 } });
    }
    /**
     * @brief The frame of the Stokes parameters that the film records, of light arriving at the focal plane along
     *        -d (telescope frame): the film's x and y, perpendicular to d. Q is light polarized along the film's x
     *        less along its y; U along x+y less along x-y.
     */
    [[nodiscard]] frame_t detector_frame(const glm::dvec3& d) const noexcept;

public:
    telescope_t(const wt_context_t &ctx,
                std::string id,
                std::shared_ptr<const optics::telescope_optics_t> optics,
                focal_plane_t focal_plane,
                placement_t placement,
                properties_t properties,
                std::vector<optics::cone_t> aim_grid,
                film_t film,
                std::uint32_t samples_per_element,
                bool ray_trace,
                f_t tan_alpha,
                f_t phase_space_extent_scale) noexcept
        : sensor::film_backed_sensor_t<2,polarimetric>(
                ctx,
                std::move(id),
                std::move(film),
                samples_per_element,
                ray_trace),
          optics(std::move(optics)),
          fp(focal_plane),
          place(placement),
          props(properties),
          aim_grid(std::move(aim_grid)),
          tan_alpha(tan_alpha),
          phase_space_extent_scale(phase_space_extent_scale)
    {}
    telescope_t(telescope_t&&) = default;

    [[nodiscard]] inline const auto& telescope_optics() const noexcept { return *optics; }
    [[nodiscard]] inline const auto& properties() const noexcept { return props; }

    [[nodiscard]] bool is_delta_position() const noexcept override  { return false; }
    [[nodiscard]] bool is_delta_direction() const noexcept override { return false; }

    /**
     * @brief Only sensor samples (backward transport): no direct connections to scene points.
     */
    [[nodiscard]] sensor::sensor_write_flags_e supported_write_flags() const noexcept override {
        return sensor::sensor_write_flags_e::writes_block_splats;
    }

    [[nodiscard]] inline beam::sourcing_geometry_t sourcing_geometry(const wavenumber_t k) const noexcept {
        const auto se =
            beam::sourcing_geometry_t::source_mub_from(tan_alpha, k)
                .phase_space_extent().enlarge(phase_space_extent_scale);
        return beam::sourcing_geometry_t::source(se);
    }
    [[nodiscard]] beam::phase_space_extent_t sourcing_beam_extent(const wavenumber_t k) const noexcept override {
        return sourcing_geometry(k).phase_space_extent();
    }

    [[nodiscard]] sensor_sample_t sample(sampler::sampler_t& sampler,
                                         const vec3u32_t& sensor_element,
                                         const wavenumber_t k) const noexcept override;

    /**
     * @brief Unsupported (see `supported_write_flags()`): returns a sample of no importance.
     */
    [[nodiscard]] sensor_direct_sample_t sample_direct(sampler::sampler_t& sampler,
                                                       const pqvec3_t& wp,
                                                       const wavenumber_t k) const noexcept override;
    /**
     * @brief Unsupported (see `supported_write_flags()`): returns 0.
     */
    [[nodiscard]] area_sampling_pd_t pdf_position(const pqvec3_t& p) const noexcept override {
        return area_sampling_pd_t{ area_density_t::zero() };
    }
    /**
     * @brief Unsupported (see `supported_write_flags()`): returns 0.
     */
    [[nodiscard]] solid_angle_sampling_pd_t pdf_direction(const pqvec3_t& p, const dir3_t& dir) const noexcept override {
        return solid_angle_density_t::zero();
    }

    [[nodiscard]] scene::element::info_t description() const override;
};

/**
 * @brief Loads a `<sensor type="telescope">`.
 */
std::shared_ptr<sensor::sensor_t> load_telescope(std::string id,
                                                 scene::loader::loader_t* loader,
                                                 const scene::loader::node_t& node,
                                                 const wt::wt_context_t &context);

}
