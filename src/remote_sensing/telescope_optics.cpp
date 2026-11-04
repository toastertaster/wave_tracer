/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

#include <wt/remote_sensing/telescope_optics.hpp>

using namespace wt::remote_sensing::optics;

namespace {

// rays start on surfaces: nearer intersections are the surface itself
constexpr double self_intersection_epsilon = 1e-9;     // metres

constexpr int newton_iterations = 32;
constexpr double newton_tolerance = 1e-13;              // metres

inline glm::dvec3 transform_point(const glm::dmat4& m, const glm::dvec3& p) noexcept {
    return glm::dvec3{ m * glm::dvec4{ p,1 } };
}
inline glm::dvec3 transform_direction(const glm::dmat4& m, const glm::dvec3& d) noexcept {
    return glm::dvec3{ m * glm::dvec4{ d,0 } };
}

}


std::optional<double> surface_shape_t::sag(double r2) const noexcept {
    const auto arg = 1 - (1+conic)*curvature*curvature*r2;
    if (arg<=0)
        return std::nullopt;
    auto z = curvature*r2 / (1 + std::sqrt(arg));
    auto r2n = r2*r2;
    for (const auto a : aspheric) {
        z += a*r2n;
        r2n *= r2;
    }
    return z;
}

std::optional<glm::dvec3> surface_shape_t::normal(double x, double y) const noexcept {
    const auto r2 = x*x + y*y;
    const auto arg = 1 - (1+conic)*curvature*curvature*r2;
    if (arg<=0)
        return std::nullopt;
    // dz/dx = x g(r²), dz/dy = y g(r²)
    auto g = curvature / std::sqrt(arg);
    auto r2n = r2;
    auto n = 4.;
    for (const auto a : aspheric) {
        g += n*a*r2n;
        r2n *= r2;
        n += 2;
    }
    return glm::normalize(glm::dvec3{ -x*g, -y*g, 1 });
}

std::optional<double> surface_shape_t::intersect(const ray_t& r, double tmin) const noexcept {
    const auto& o = r.o;
    const auto& d = r.d;

    // the base conic, c (x² + y² + (1+κ) z²) - 2z = 0: of the quadratic's roots, the one through the vertex (which
    // tends to the plane's as c tends to 0)
    const auto c = curvature, k1 = 1+conic;
    const auto A = c*(d.x*d.x + d.y*d.y + k1*d.z*d.z);
    const auto B = c*(o.x*d.x + o.y*d.y + k1*o.z*d.z) - d.z;
    const auto C = c*(o.x*o.x + o.y*o.y + k1*o.z*o.z) - 2*o.z;
    const auto D = B*B - A*C;

    double t;
    if (D>=0 && (B!=0 || A!=0)) {
        const auto q = B + std::copysign(std::sqrt(D), B);
        if (q==0)
            return std::nullopt;
        t = -C/q;
    } else if (aspheric.empty()) {
        return std::nullopt;
    } else {
        // the aspheric terms may still bring the surface to the ray: start at the vertex plane
        if (d.z==0)
            return std::nullopt;
        t = -o.z/d.z;
    }
    if (!std::isfinite(t))
        return std::nullopt;

    // refine on the full surface (for a bare conic: polishes the root)
    bool converged = false;
    for (int i=0; i<newton_iterations; ++i) {
        const auto p = o + t*d;
        const auto z = sag(p.x*p.x + p.y*p.y);
        const auto n = normal(p.x, p.y);
        if (!z || !n)
            return std::nullopt;
        // F(t) = z(t) - sag(x(t),y(t)), F' = d · (-dz/dx, -dz/dy, 1) = d · n / n.z
        const auto dF = glm::dot(d, *n) / n->z;
        if (dF==0)
            return std::nullopt;
        const auto dt = (p.z - *z) / dF;
        t -= dt;
        if (std::abs(dt)<=newton_tolerance) {
            converged = true;
            break;
        }
    }
    if (!converged || !(t>tmin))
        return std::nullopt;
    return t;
}


std::optional<element_t::hit_t> element_t::intersect(const ray_t& r, double tmin) const noexcept {
    const auto local = ray_t{ transform_point(to_local, r.o), transform_direction(to_local, r.d) };
    const auto t = shape.intersect(local, tmin);
    if (!t)
        return std::nullopt;
    return hit_t{ .t = *t, .local = local.o + *t*local.d };
}


glm::dvec3 cone_t::sample(double u1, double u2) const noexcept {
    const auto cos_theta = 1 - u1*(1-cos_half_angle);
    const auto sin_theta = std::sqrt(std::max(0., 1 - cos_theta*cos_theta));
    const auto phi = 2*std::numbers::pi*u2;
    const auto helper = std::abs(axis.x)<.9 ? glm::dvec3{ 1,0,0 } : glm::dvec3{ 0,1,0 };
    const auto t = glm::normalize(glm::cross(helper, axis));
    const auto b = glm::cross(axis, t);
    return glm::normalize(cos_theta*axis + sin_theta*(std::cos(phi)*t + std::sin(phi)*b));
}


telescope_optics_t::telescope_optics_t(std::vector<element_t> elements)
    : elems(std::move(elements))
{
    for (std::size_t i=0; i<elems.size(); ++i) {
        const auto& e = elems[i];
        const auto name = e.id.empty() ? std::format("#{}", i) : "<" + e.id + ">";
        if (!(e.aperture.radius>0))
            throw std::invalid_argument(std::format("element {}: the aperture's radius must be positive", name));
        if (!(e.aperture.hole_radius>=0) || !(e.aperture.hole_radius<e.aperture.radius))
            throw std::invalid_argument(std::format("element {}: the hole's radius must be in [0, aperture radius)", name));
        if (e.kind!=element_kind_e::mirror && (e.shape.curvature!=0 || !e.shape.aspheric.empty()))
            throw std::invalid_argument(std::format("element {}: stops and obscurations are flat", name));
        if (!e.shape.sag(glm::dot(e.aperture.centre, e.aperture.centre) +
                         e.aperture.radius*(e.aperture.radius + 2*glm::length(e.aperture.centre))))
            throw std::invalid_argument(std::format("element {}: the aperture extends beyond the conic", name));
        if (e.kind!=element_kind_e::obscuration)
            seq.push_back(i);
    }
    const auto mirrors = std::ranges::count_if(elems, [](const auto& e) { return e.kind==element_kind_e::mirror; });
    if (mirrors==0)
        throw std::invalid_argument("at least one mirror is needed");
    if (std::size_t(mirrors)>max_mirrors)
        throw std::invalid_argument(std::format("at most {} mirrors are supported", max_mirrors));

    // bound the clear aperture nearest the focal plane by a sphere about its centre
    const auto& last = elems[seq.back()];
    const auto& a = last.aperture;
    const auto centre = glm::dvec3{ a.centre, *last.shape.sag(glm::dot(a.centre,a.centre)) };
    double radius = 0;
    constexpr int rim_points = 360;
    for (int i=0; i<rim_points; ++i) {
        const auto phi = 2*std::numbers::pi*i/rim_points;
        const auto xy = a.centre + a.radius*glm::dvec2{ std::cos(phi), std::sin(phi) };
        const auto rim = glm::dvec3{ xy, *last.shape.sag(glm::dot(xy,xy)) };
        radius = std::max(radius, glm::length(rim - centre));
    }
    aim_centre = transform_point(last.to_telescope, centre);
    // margin for the rim between the points
    aim_radius = radius * 1.001;
}

std::size_t telescope_optics_t::obstacle(const ray_t& r, double tmax,
                                         std::size_t from, std::size_t to) const noexcept {
    for (std::size_t i=0; i<elems.size(); ++i) {
        const auto& e = elems[i];
        if (i==from || i==to || e.kind==element_kind_e::stop)
            continue;
        const auto hit = e.intersect(r, self_intersection_epsilon);
        if (hit && hit->t<tmax-self_intersection_epsilon && e.aperture.contains(hit->local.x, hit->local.y))
            return i;
    }
    return npos;
}

telescope_optics_t::trace_t telescope_optics_t::trace_from_focal_plane(const ray_t& r) const noexcept {
    trace_t ret;
    const auto stopped = [&ret](std::size_t by) {
        ret.stopped_by = by;
        return ret;
    };

    auto ray = r;
    auto from = npos;
    for (auto it = seq.rbegin(); it!=seq.rend(); ++it) {
        const auto idx = *it;
        const auto& e = elems[idx];

        const auto hit = e.intersect(ray, from==npos ? 0 : self_intersection_epsilon);
        if (!hit || !e.aperture.contains(hit->local.x, hit->local.y))
            return stopped(idx);
        if (const auto o = obstacle(ray, hit->t, from, idx); o!=npos)
            return stopped(o);

        const auto p = ray.o + hit->t*ray.d;
        auto d = ray.d;
        if (e.kind==element_kind_e::mirror) {
            const auto n_local = e.shape.normal(hit->local.x, hit->local.y);
            if (!n_local)
                return stopped(idx);
            auto n = glm::normalize(transform_direction(e.to_telescope, *n_local));
            if (glm::dot(d,n)>0)
                n = -n;
            d = glm::normalize(d - 2*glm::dot(d,n)*n);
            ret.reflections[ret.reflection_count++] = { .element = idx, .normal = n, .d_in = ray.d, .d_out = d };
        }
        ray = ray_t{ p,d };
        from = idx;
    }

    // out towards the scene
    if (const auto o = obstacle(ray, std::numeric_limits<double>::infinity(), from, npos); o!=npos)
        return stopped(o);
    ret.passed = true;
    ret.exit = ray;
    return ret;
}

cone_t telescope_optics_t::aim(const glm::dvec3& p) const noexcept {
    const auto v = aim_centre - p;
    const auto dist = glm::length(v);
    if (dist<=aim_radius)
        return { .axis = { 0,0,-1 }, .cos_half_angle = -1 };     // all directions
    const auto s = aim_radius/dist;
    return { .axis = v/dist, .cos_half_angle = std::sqrt(1 - s*s) };
}

field_point_t telescope_optics_t::analyse(const glm::dvec3& p, const glm::dvec3& normal,
                                          std::uint32_t samples) const noexcept {
    const auto cone = aim(p);
    const auto N = double(samples);

    std::vector<ray_t> exits;
    std::vector<glm::dvec3> passing;
    std::vector<double> weights;
    double psa = 0;
    glm::dvec3 los = { 0,0,0 };
    // a Fibonacci lattice over the cone: every sample at its own angle from the axis, so that apertures' edges (at
    // constant angles, on axis) cost an error of order 1/N rather than a grid's 1/√N
    constexpr double golden = 0.6180339887498949;
    for (std::uint32_t i=0; i<samples; ++i) {
        const auto d = cone.sample((i+.5)/N, m::fract(golden*i));
        const auto tr = trace_from_focal_plane({ p,d });
        if (!tr.passed)
            continue;
        const auto w = std::abs(glm::dot(d, normal));
        psa += w;
        los += w*tr.exit.d;
        exits.push_back(tr.exit);
        passing.push_back(d);
        weights.push_back(w);
    }

    field_point_t ret;
    if (exits.empty())
        return ret;
    ret.passed_fraction = exits.size()/N;
    ret.projected_solid_angle = psa * cone.solid_angle()/N;
    ret.line_of_sight = glm::normalize(los);

    double spread2 = 0;
    glm::dvec3 centroid = { 0,0,0 };
    for (std::size_t i=0; i<exits.size(); ++i) {
        const auto& d = exits[i].d;
        const auto angle = std::atan2(glm::length(glm::cross(d, ret.line_of_sight)), glm::dot(d, ret.line_of_sight));
        spread2 += weights[i]*angle*angle;
        centroid += exits[i].o - glm::dot(exits[i].o, ret.line_of_sight)*ret.line_of_sight;
    }
    ret.rms_spread = std::sqrt(spread2/psa);
    centroid /= double(exits.size());

    // the bundle's cross section, perpendicular to the line of sight
    double rmin = std::numeric_limits<double>::infinity(), rmax = 0;
    for (const auto& e : exits) {
        const auto q = e.o - glm::dot(e.o, ret.line_of_sight)*ret.line_of_sight;
        const auto r = glm::length(q - centroid);
        rmin = std::min(rmin, r);
        rmax = std::max(rmax, r);
    }
    ret.pupil_diameter = 2*rmax;
    ret.pupil_obscuration = rmax>0 ? rmin/rmax : 0;

    glm::dvec3 axis = { 0,0,0 };
    for (const auto& d : passing)
        axis += d;
    axis = glm::normalize(axis);
    double cos_half_angle = 1;
    for (const auto& d : passing)
        cos_half_angle = std::min(cos_half_angle, glm::dot(d, axis));
    ret.passing_cone = { .axis = axis, .cos_half_angle = cos_half_angle };

    return ret;
}
