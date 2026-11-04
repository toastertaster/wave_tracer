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
focal plane. Only centred two-mirror designs are modelled.

    uv run tools/remote_sensing/telescope_reference.py

prints the values, as JSON, for the Rust bindings' tests (tests/remote_sensing.rs).
"""

import json
import math

import numpy as np
from optiland import optic, physical_apertures

from telescope import hubble, two_mirror

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
    print(json.dumps(refs, indent=2))


if __name__ == "__main__":
    main()
