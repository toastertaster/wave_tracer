# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "optiland"]
# ///
"""Reference values for the telescope sensor's tests, computed by Optiland (an independent sequential ray tracer) for
telescope.py's designs: the paraxial focal length, the entrance pupil, and the relative throughput of a point on the
focal plane off the axis (vignetting and relative illumination).

The throughput of a focal-plane point is the projected solid angle of the light that reaches it. By the conservation
of étendue, at the image of field angle θ (at height h(θ) from the axis, the image's centroid), it is
    A(θ) cos θ sin θ / (h h'),
A(θ) the entrance pupil's area whose rays reach the focal plane (an area perpendicular to the axis, whose beam's
cross section is A cos θ), and sin θ dθ dφ / (h dh dφ) the solid angle per focal-plane area. On axis it is A(0)/f²;
without distortion (h = f tan θ), the relative throughput is A(θ)/A(0) cos⁴ θ. Distortion stretches the image, and
dims it as much.

Each design is modelled in Optiland as the telescope sensor sees it: light meets the secondary's plane first, where the
secondary's body (a disc of its aperture radius) blocks it; then the primary (the stop: its aperture and central
hole), the secondary (its aperture), and the primary's plane again, whose hole the converging beam passes; then a flat
focal plane. Only centred two-mirror designs are modelled so.

Off-axis segments of coaxial parents (the three-mirror anastigmat; the off-axis Ritchey-Chrétien, on its flat focal
plane and on the sphere telescope.py's --curved-focal-plane fits to its field) are modelled as their parents, whose
entrance pupil covers the segment; their rays are traced through the segment only (normalized pupil coordinates within
it), at the field that images at the film's centre, onto the image surface (a curved focal plane's sphere: centred on
the axis), in the focal plane's x and y. From them: the parent's focal length; the image's local scale (its centroid's
motion per field angle, along x and y); the RMS spot there; and the film centre's projected solid angle that the
étendue gives, A cos θ dΩ/dA (the segment's area, seen from the field direction θ, times the solid angle per image
area: the Jacobian of the field's directions to the image's centroids). The segment's area is the sensor's: of a disc on
the primary's surface, which the field's rays cross the pupil's plane over.

    uv run tools/remote_sensing/telescope_reference.py

prints the values, as JSON, for the Rust bindings' tests (tests/remote_sensing.rs).
"""

import json
import math

import numpy as np
from optiland import optic, physical_apertures

from telescope import curve_focal_plane, hubble, three_mirror, two_mirror

INF = float("inf")


def optiland_model(design, max_field_deg=0.0):
    """An Optiland model of a centred two-mirror design (millimetres), with fields on axis and at
    `max_field_deg` along y."""
    primary, secondary = design.elements[:2]
    if any(primary.aperture_centre) or any(secondary.aperture_centre) or len(design.elements) != 2:
        raise ValueError("only centred two-mirror designs are modelled")
    mm = 1e3
    d = -secondary.to_telescope[2, 3] * mm
    b = design.focal_plane[2, 3] * mm
    hole = primary.hole_radius * mm

    o = optic.Optic()
    o.surfaces.add(index=0, radius=INF, thickness=INF)
    # the secondary's plane: its body's shadow
    o.surfaces.add(index=1, thickness=d,
                   aperture=physical_apertures.RadialAperture(r_max=INF, r_min=secondary.aperture_radius * mm))
    o.surfaces.add(index=2, radius=primary.radius * mm, conic=primary.conic, thickness=-d, material="mirror",
                   is_stop=True,
                   aperture=physical_apertures.RadialAperture(r_max=primary.aperture_radius * mm, r_min=hole))
    o.surfaces.add(index=3, radius=secondary.radius * mm, conic=secondary.conic, thickness=d, material="mirror",
                   aperture=physical_apertures.RadialAperture(r_max=secondary.aperture_radius * mm))
    # the primary's plane: the converging beam passes its hole
    o.surfaces.add(index=4, thickness=b,
                   aperture=physical_apertures.RadialAperture(r_max=hole if hole > 0 else INF))
    o.surfaces.add(index=5)
    o.set_aperture(aperture_type="EPD", value=2 * primary.aperture_radius * mm)
    o.fields.set_type(field_type="angle")
    o.fields.add(y=0)
    if max_field_deg:
        o.fields.add(y=max_field_deg)
    o.wavelengths.add(value=0.55, is_primary=True)
    return o


def trace(o, Hy, rays):
    """At normalized field Hy (along y): the fraction of a uniform grid over the entrance pupil whose rays reach the
    focal plane, and their centroid's height there (mm)."""
    r = o.trace(0, Hy, 0.55, rays, "uniform")
    passed = np.asarray(r.i) > 0
    return float(np.count_nonzero(passed)) / len(passed), float(np.mean(np.asarray(r.y)[passed]))


def relative_throughput(design, image_height, rays=2048):
    """The throughput of the focal-plane point `image_height` (m) off the axis, relative to the axis'."""
    f = design.focal_length
    theta_max = math.atan(image_height / f) * 1.01
    o = optiland_model(design, math.degrees(theta_max))
    height = lambda theta: abs(trace(o, theta / theta_max, rays)[1]) / 1e3
    # the field angle that images there (Newton, from the paraxial one)
    theta = math.atan(image_height / f)
    for _ in range(4):
        theta -= (height(theta) - image_height) / f
    dtheta = 1e-3 * theta
    dh = (height(theta + dtheta) - height(theta - dtheta)) / (2 * dtheta)
    h = height(theta)
    area = trace(o, theta / theta_max, rays)[0] / trace(o, 0, rays)[0]
    return {
        "image_height_m": image_height,
        "field_deg": math.degrees(theta),
        "pupil_fraction": area,
        "relative_throughput": area * math.cos(theta) * math.sin(theta) * f * f / (h * dh),
    }


def reference(design):
    o = optiland_model(design)
    return {
        "focal_length_m": float(o.paraxial.f2()) / 1e3,
        "entrance_pupil_diameter_m": float(o.paraxial.EPD()) / 1e3,
    }
    return out


def segment_reference(design, field_bias, off_axis):
    """Optiland's values for an off-axis segment of telescope.py's coaxial mirrors (the three-mirror anastigmat, or a
    two-mirror design's segment), at the field that images at the film's centre (from `field_bias`: radians, along y,
    the light's direction of travel), on the design's focal plane (see the module's doc)."""
    mm = 1e3
    mirrors = design.elements
    primary = mirrors[0]
    z = [e.to_telescope[2, 3] for e in mirrors]
    D = 2 * primary.aperture_radius
    epd = 2 * (off_axis + D / 2)

    # the image surface: the focal plane, or the sphere it is curved on (centred on the axis), from its vertex
    fp = design.focal_plane
    origin, x_axis, y_axis, z_axis = fp[:3, 3], fp[:3, 0], fp[:3, 1], fp[:3, 2]
    if design.focal_plane_radius:
        centre = origin + design.focal_plane_radius * z_axis
        if np.hypot(centre[0], centre[1]) > 1e-9:
            raise ValueError("the curved focal plane's centre is off the axis")
        r = abs(design.focal_plane_radius)
        z_image = min(centre[2] - r, centre[2] + r, key=lambda v: abs(v - origin[2]))
        image_radius = centre[2] - z_image
    else:
        if abs(abs(z_axis[2]) - 1) > 1e-12:
            raise ValueError("a flat focal plane is modelled perpendicular to the axis")
        z_image, image_radius = origin[2], INF

    o = optic.Optic()
    o.surfaces.add(index=0, radius=INF, thickness=INF)
    for i, e in enumerate(mirrors):
        z_next = z[i + 1] if i + 1 < len(mirrors) else z_image
        o.surfaces.add(index=i + 1, radius=e.radius * mm, conic=e.conic, thickness=(z_next - z[i]) * mm,
                       material="mirror", is_stop=i == 0)
    o.surfaces.add(index=len(mirrors) + 1, radius=image_radius * mm)
    o.set_aperture(aperture_type="EPD", value=epd * mm)
    o.fields.set_type(field_type="angle")
    max_field = abs(math.degrees(field_bias)) * 2
    o.fields.add(y=0)
    o.fields.add(y=max_field)
    o.wavelengths.add(value=0.55, is_primary=True)

    # the segment's rays: a polar grid, uniform by area, in normalized pupil coordinates
    n_r, n_phi = 40, 72
    rr = np.sqrt((np.arange(n_r) + 0.5) / n_r) * (D / 2)
    phi = 2 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    R, PHI = np.meshgrid(rr, phi)
    px = (R * np.cos(PHI)).ravel() / (epd / 2)
    py = (off_axis + R * np.sin(PHI)).ravel() / (epd / 2)

    def image(theta_x, theta_y):
        """The rays' points on the image surface, in the focal plane's x and y (about its origin)."""
        hx, hy = np.full_like(px, math.degrees(theta_x) / max_field), np.full_like(py, math.degrees(theta_y) / max_field)
        o.trace_generic(hx, hy, px, py, 0.55)
        # (Optiland's global coordinates: the telescope's, the first mirror's vertex at the origin)
        p = np.stack([np.asarray(o.surfaces.x[-1]), np.asarray(o.surfaces.y[-1]),
                      np.asarray(o.surfaces.z[-1])], axis=1) / mm
        return (p - origin) @ x_axis, (p - origin) @ y_axis

    # the field that images at the film's centre (the focal plane's origin), from `field_bias` (Newton)
    delta = 2e-5
    for _ in range(4):
        slope = (image(0.0, field_bias + delta)[1].mean() - image(0.0, field_bias - delta)[1].mean()) / (2 * delta)
        field_bias -= image(0.0, field_bias)[1].mean() / slope

    x0, y0 = image(0.0, field_bias)
    rms = float(np.sqrt(np.mean((x0 - x0.mean()) ** 2 + (y0 - y0.mean()) ** 2)))
    xp, yp = image(delta, field_bias)
    xm, ym = image(-delta, field_bias)
    xq, yq = image(0.0, field_bias + delta)
    xr, yr = image(0.0, field_bias - delta)
    J = np.array([[(xp.mean() - xm.mean()), (xq.mean() - xr.mean())],
                  [(yp.mean() - ym.mean()), (yq.mean() - yr.mean())]]) / (2 * delta)
    # the field's direction, (tan θx, tan θy, 1) normalized: the angles its parameters turn it by, and its solid
    # angle per dθx dθy
    def direction(tx, ty):
        v = np.array([math.tan(tx), math.tan(ty), 1.0])
        return v / np.linalg.norm(v)
    dx = (direction(delta, field_bias) - direction(-delta, field_bias)) / (2 * delta)
    dy = (direction(0, field_bias + delta) - direction(0, field_bias - delta)) / (2 * delta)
    d_omega = np.linalg.norm(np.cross(dx, dy))
    # the clear aperture is a disc on the primary's surface: the field's rays through it cross the plane z = 0 over
    # its area times 1 - t·∇sag (t the rays' slopes), 0.29% more than the disc's at a 1° bias
    c, k = 1 / primary.radius, primary.conic
    g = np.linspace(-1, 1, 801)
    gx, gy = np.meshgrid(g, g)
    inside = gx ** 2 + gy ** 2 <= 1
    x, y = (D / 2) * gx[inside], off_axis + (D / 2) * gy[inside]
    slope = c / np.sqrt(1 - (1 + k) * c * c * (x * x + y * y))       # ∇sag = (x, y) slope
    area = math.pi * (D / 2) ** 2 * float(np.mean(1 - math.tan(field_bias) * y * slope))
    cos_theta = direction(0, field_bias)[2]
    return {
        "focal_length_m": float(o.paraxial.f2()) / mm,
        # per angle between the directions (as the sensor's focal lengths: from its lines of sight)
        "scale_x_m_per_rad": float(abs(J[0, 0]) / np.linalg.norm(dx)),
        "scale_y_m_per_rad": float(abs(J[1, 1]) / np.linalg.norm(dy)),
        "rms_spot_m": rms,
        "field_deg": math.degrees(field_bias),
        "projected_solid_angle_sr": float(area * cos_theta * d_omega / abs(np.linalg.det(J))),
    }


def main():
    pixel_pitch, pixels = 5e-6, 256
    field = math.atan(pixels * pixel_pitch / math.sqrt(2) / 2.5)
    designs = {
        "cassegrain": two_mirror("cassegrain", field_radius=field),
        "ritchey_chretien": two_mirror("ritchey-chretien", field_radius=field),
        "hubble": hubble(),
    }
    refs = {name: reference(d) for name, d in designs.items()}

    # vignetting: an undersized secondary (its aperture the on-axis beam's footprint, k D/2, plus 0.2 mm), and the
    # corner of a 64² film of 0.5 mm pixels, 0.51° off axis (telescope.py's defaults: D 0.4 m, f 2.5 m, f1 0.6 m,
    # b 0.15 m)
    vig_pitch, vig_pixels = 500e-6, 64
    k = (1 + 0.15 / 0.6) / (1 + 2.5 / 0.6)
    secondary_radius = k * 0.4 / 2 + 0.2e-3
    vig = two_mirror("ritchey-chretien", field_radius=math.atan(vig_pixels * vig_pitch / math.sqrt(2) / 2.5),
                     secondary_radius=secondary_radius)
    # the corner pixels' centres
    corner = (vig_pixels / 2 - 0.5) * vig_pitch * math.sqrt(2)
    refs["vignetting"] = (reference(vig) | relative_throughput(vig, corner)
                          | {"secondary_radius_m": secondary_radius})
    bias, off_axis = math.radians(1.0), 0.2
    tma = three_mirror(0.2, 2.5, 0.6, (0.45, 1.15), off_axis=off_axis, field_bias=bias,
                       field_radius=math.atan(pixels * pixel_pitch / math.sqrt(2) / 2.5))
    refs["three_mirror"] = segment_reference(tma, bias, off_axis)

    # the off-axis Ritchey-Chrétien segment, 0.3° off its axis (north: light travelling south), on its flat focal
    # plane and on the sphere telescope.py fits to its field
    field = math.atan(pixels * pixel_pitch / math.sqrt(2) / 2.5)
    segment = two_mirror("ritchey-chretien", 0.2, field_radius=field, off_axis=0.2, field_bias=math.radians(0.3))
    refs["off_axis_biased"] = segment_reference(segment, -math.radians(0.3), 0.2)
    refs["off_axis_curved"] = segment_reference(curve_focal_plane(segment, field), -math.radians(0.3), 0.2)
    print(json.dumps(refs, indent=2))


if __name__ == "__main__":
    main()
