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
#include <wt/sampler/sampler.hpp>

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
    if (kind==element_kind_e::tube) {
        // x² + y² = R², the nearer root beyond tmin within the length
        const auto& o = local.o;
        const auto& d = local.d;
        const auto A = d.x*d.x + d.y*d.y;
        if (A==0)
            return std::nullopt;
        const auto B = o.x*d.x + o.y*d.y;
        const auto C = o.x*o.x + o.y*o.y - aperture.radius*aperture.radius;
        const auto D = B*B - A*C;
        if (D<0)
            return std::nullopt;
        const auto q = -(B + std::copysign(std::sqrt(D), B));
        auto t0 = q/A, t1 = q!=0 ? C/q : t0;
        if (t0>t1)
            std::swap(t0, t1);
        for (const auto t : { t0, t1 }) {
            const auto z = o.z + t*d.z;
            if (t>tmin && z>=0 && z<=length)
                return hit_t{ .t = t, .local = o + t*d };
        }
        return std::nullopt;
    }
    const auto t = shape.intersect(local, tmin);
    if (!t)
        return std::nullopt;
    return hit_t{ .t = *t, .local = local.o + *t*local.d };
}

std::optional<element_t::hit_t> element_t::physical_hit(const ray_t& r, double tmin) const noexcept {
    const auto hit = intersect(r, tmin);
    if (!hit)
        return std::nullopt;
    const bool within = aperture.contains(hit->local.x, hit->local.y);
    switch (kind) {
    case element_kind_e::tube:  return hit;
    case element_kind_e::stop:  return within ? std::nullopt : hit;
    default:                    return within ? hit : std::nullopt;
    }
}

glm::dvec3 element_t::normal_at(const glm::dvec3& local) const noexcept {
    glm::dvec3 n = { 0,0,1 };
    if (kind==element_kind_e::tube)
        n = glm::dvec3{ local.x, local.y, 0 };
    else if (kind==element_kind_e::mirror)
        n = shape.normal(local.x, local.y).value_or(n);
    return glm::normalize(transform_direction(to_telescope, n));
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
            throw std::invalid_argument(std::format("element {}: stops, obscurations and tubes are flat", name));
        if (e.kind==element_kind_e::tube && !(e.length>0))
            throw std::invalid_argument(std::format("element {}: a tube's length must be positive", name));
        if (!(e.reflectance>=0 && e.reflectance<=1))
            throw std::invalid_argument(std::format("element {}: the reflectance must be in [0,1]", name));
        if (e.kind!=element_kind_e::mirror && e.scatter)
            throw std::invalid_argument(std::format("element {}: only mirrors scatter (ABg)", name));
        if (!e.shape.sag(glm::dot(e.aperture.centre, e.aperture.centre) +
                         e.aperture.radius*(e.aperture.radius + 2*glm::length(e.aperture.centre))))
            throw std::invalid_argument(std::format("element {}: the aperture extends beyond the conic", name));
        if (e.kind==element_kind_e::mirror || e.kind==element_kind_e::stop)
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
        const auto hit = e.physical_hit(r, self_intersection_epsilon);
        if (hit && hit->t<tmax-self_intersection_epsilon)
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
            // a rough mirror scatters some of the light out of the image: |β₀| = sin θᵢ
            if (e.scatter) {
                const auto c = glm::dot(d,n);
                ret.weight *= 1 - e.scatter->tis(std::sqrt(std::max(0., 1 - c*c)));
            }
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


void telescope_optics_t::orient(const trace_t& designed) noexcept {
    for (std::uint32_t i=0; i<designed.reflection_count; ++i) {
        const auto& r = designed.reflections[i];
        auto& e = elems[r.element];
        // the side `r.normal` (facing the ray) is on: the element's local +z side, or not (its surface's normals
        // facing local +z are within 90° of local +z)
        const auto lz = glm::normalize(transform_direction(e.to_telescope, { 0,0,1 }));
        e.reflective_side = glm::dot(r.normal, lz)>=0 ? 1 : -1;
    }
}

std::size_t telescope_optics_t::nearest(const ray_t& r, double tmin, const detector_t& det,
                                        double& t, glm::dvec3& local) const noexcept {
    auto found = npos;
    t = std::numeric_limits<double>::infinity();
    for (std::size_t i=0; i<elems.size(); ++i) {
        const auto hit = elems[i].physical_hit(r, tmin);
        if (hit && hit->t<t) {
            t = hit->t;
            local = hit->local;
            found = i;
        }
    }
    // the detector's plane
    const auto o = transform_point(det.to_local, r.o);
    const auto d = transform_direction(det.to_local, r.d);
    if (d.z!=0) {
        const auto td = -o.z/d.z;
        const auto p = o + td*d;
        if (td>tmin && td<t && std::abs(p.x)<=det.half_extent.x && std::abs(p.y)<=det.half_extent.y) {
            t = td;
            found = stopped_by_detector;
        }
    }
    return found;
}

bool telescope_optics_t::escapes(const ray_t& r, double tmin, const detector_t& det) const noexcept {
    double t;
    glm::dvec3 local;
    return nearest(r, tmin, det, t, local)==npos;
}

namespace {

// an orthonormal tangent frame about unit n
void tangent_frame(const glm::dvec3& n, glm::dvec3& t, glm::dvec3& b) noexcept {
    const auto helper = std::abs(n.x)<.9 ? glm::dvec3{ 1,0,0 } : glm::dvec3{ 0,1,0 };
    t = glm::normalize(glm::cross(helper, n));
    b = glm::cross(n, t);
}

}

telescope_optics_t::trace_t telescope_optics_t::trace_nonsequential(
        const ray_t& r, const detector_t& det, double scatter_probability,
        sampler::sampler_t& sampler,
        const std::function<void(const scatter_point_t&)>* at_scatter) const {
    trace_t ret;
    auto ray = r;
    while (true) {
        double t;
        glm::dvec3 local;
        const auto idx = nearest(ray, self_intersection_epsilon, det, t, local);
        if (idx==npos) {
            ret.passed = true;
            ret.exit = ray;
            return ret;
        }
        ret.stopped_by = idx;
        if (idx==stopped_by_detector || ret.reflection_count==max_events)
            return ret;

        const auto& e = elems[idx];
        const auto p = ray.o + t*ray.d;
        auto n = e.normal_at(local);
        const bool against = glm::dot(ray.d, n)<0;     // the ray meets the side n faces
        if (!against)
            n = -n;
        auto event = reflection_t{ .element = idx, .normal = n, .d_in = ray.d, .d_out = ray.d };

        if (e.kind==element_kind_e::mirror) {
            // the back of a mirror absorbs
            if (e.reflective_side!=0 && (e.reflective_side>0)!=against)
                return ret;
            const auto spec = glm::normalize(ray.d - 2*glm::dot(ray.d,n)*n);
            if (!e.scatter) {
                event.d_out = spec;
            } else {
                if (at_scatter)
                    (*at_scatter)({ .p = p, .d_in = ray.d, .element = idx, .normal = n, .so_far = ret });
                const auto c = glm::dot(spec, n);
                const auto beta0_len = std::sqrt(std::max(0., 1 - c*c));
                const auto tis = e.scatter->tis(beta0_len);
                const auto p_scatter = std::clamp(std::max(tis, scatter_probability), 0., 1.);
                if (double(sampler.r())>=p_scatter) {
                    event.d_out = spec;
                    ret.weight *= (1-tis) / (1-p_scatter);
                } else {
                    // about the specular direction, in its projection on the tangent plane
                    glm::dvec3 tt, bb;
                    tangent_frame(n, tt, bb);
                    const auto beta0 = glm::dvec2{ glm::dot(spec,tt), glm::dot(spec,bb) };
                    const auto u = sampler.r2();
                    const auto smp = e.scatter->sample(double(sampler.r()), double(u.x), double(u.y));
                    const auto beta = beta0 + smp.offset;
                    const auto b2 = glm::dot(beta, beta);
                    if (!(b2<1) || smp.recp_pdf==0)
                        return ret;         // evanescent: nothing scattered there
                    event.d_out = glm::normalize(beta.x*tt + beta.y*bb + std::sqrt(1-b2)*n);
                    event.kind = event_kind_e::scattered;
                    ret.weight *= e.scatter->bsdf(glm::length(smp.offset)) * smp.recp_pdf / p_scatter;
                }
            }
        } else {
            // a Lambertian reflection, of its reflectance (sampled)
            if (!(e.reflectance>0))
                return ret;
            if (at_scatter)
                (*at_scatter)({ .p = p, .d_in = ray.d, .element = idx, .normal = n, .so_far = ret });
            if (double(sampler.r())>=e.reflectance)
                return ret;
            const auto local_dir = sampler::sampler_t::cosine_hemisphere(sampler.r2());
            glm::dvec3 tt, bb;
            tangent_frame(n, tt, bb);
            event.d_out = glm::normalize(double(local_dir.x)*tt + double(local_dir.y)*bb + double(local_dir.z)*n);
            event.kind = event_kind_e::diffuse;
        }

        ret.reflections[ret.reflection_count++] = event;
        ray = ray_t{ p, event.d_out };
    }
}
