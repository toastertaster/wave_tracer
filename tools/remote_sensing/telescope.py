# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Telescope designs for the telescope sensor: writes a <sensor type="telescope"> of a design.

The telescope sensor (see the scene elements' docs) traces its rays through the mirrors that its scene element lists;
this tool computes those elements for a few kinds of designs from a handful of parameters, so that scenes need not be
written by hand:

  two-mirror   a Cassegrain telescope from its first-order layout: aperture diameter D, focal length f, primary
               focal length f1 and back focus b (from the primary's vertex to the focal plane, behind it). The
               secondary's magnification is m = f/f1, its distance from the primary d = f1 (1 - k) with
               k = (1 + b/f1)/(1 + m) the ratio of the marginal ray's heights on the mirrors, and its radius of
               curvature 2 k f1 m/(m - 1) (Schroeder, Astronomical Optics, ch. 6). Mirror shapes (--kind):
                 cassegrain        classical: a paraboloid primary (conic -1), a hyperboloid secondary
                                   (conic -((m+1)/(m-1))²)
                 ritchey-chretien  aplanatic (free of coma as well as of spherical aberration), with
                                   β = b/f1: primary conic -1 - 2(1+β)/(m²(m-β)), secondary conic
                                   -((m+1)/(m-1))² - 2m(m+1)/((m-β)(m-1)³)
               The secondary's clear aperture covers the field's beams, and the primary's central hole passes the
               beam converging on the focal plane; the secondary's shadow is the central obscuration.
               --off-axis Y uses an off-axis segment of the design instead: the primary's aperture (diameter D)
               centred Y from the axis, the secondary's on its beam, both unobscured. --field-bias puts the film's
               centre off the axis, at that field angle (towards north). --fold adds a flat mirror at 45° in the
               converging beam, that far before the focal plane, folding it towards north (+y): with three mirrors,
               the image is mirrored (east to the film's left).
  hubble       the Hubble Space Telescope's Ritchey-Chrétien (Laikin, Lens Design, 4th ed., p. 200, as in
               Optiland's sample), with a flat focal plane at its image surface's vertex.

The telescope's frame follows optical design software's convention: light arrives travelling towards +z, the primary
at the origin. The film's x is east and its y south at nadir (as a satellite_frame camera's) for designs with an even
number of mirrors. Lengths in the scene are in metres.

--coating coats every mirror (default ideal: a perfect conductor, reflecting all light):
  aluminium, silver, gold   the bare metals (wave_tracer's refractive-index data)
  protected-aluminium       aluminium under 25 nm of magnesium fluoride (n = 1.38), as Hubble's mirrors

    uv run tools/remote_sensing/telescope.py two-mirror --kind ritchey-chretien [--diameter 0.4 --focal-length 2.5 ...] > sensor.xml

prints the sensor (with --altitude, --view-zenith, ... placing it, as strings, so that scene defines like $vz work).
As a module, `two_mirror()` and `hubble()` return designs, and `sensor_xml()` writes them.
"""

import argparse
import dataclasses
import math

import numpy as np


@dataclasses.dataclass
class Element:
    """A mirror, stop or obscuration of the telescope sensor: its shape, clear aperture and placement (4×4, element
    frame to telescope frame, metres)."""
    kind: str
    id: str
    aperture_radius: float
    to_telescope: np.ndarray
    radius: float | None = None
    conic: float = 0.0
    aspheric: list[float] = dataclasses.field(default_factory=list)
    hole_radius: float = 0.0
    aperture_centre: tuple[float, float] = (0.0, 0.0)
    coating: str = "ideal"


@dataclasses.dataclass
class Design:
    """A telescope: its elements (in the order light meets them), the focal plane's frame (film to telescope), and
    its first-order properties."""
    name: str
    elements: list[Element]
    focal_plane: np.ndarray
    focal_length: float
    entrance_pupil_diameter: float
    obscuration: float


def translation(x=0.0, y=0.0, z=0.0):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def frame(origin, x, y, z):
    """The rigid transform of a frame of axes x, y, z (unit, right-handed) at origin."""
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, origin
    return m


# the film's x east and y south: the image of a two-mirror telescope is inverted, at nadir an object east of the line
# of sight images towards -x, one north towards -y; half a turn about y
UPRIGHT = frame((0, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 0, -1))


def two_mirror(kind="ritchey-chretien", diameter=0.4, focal_length=2.5, primary_focal_length=0.6, back_focus=0.15,
               field_radius=math.radians(0.1), off_axis=0.0, field_bias=0.0, fold=None,
               secondary_radius=None, margin=1e-3):
    """A Cassegrain telescope (see the module's doc). Angles in radians, lengths in metres. `field_radius` is the
    half-field whose beams the secondary's aperture and the primary's hole pass (beyond `field_bias`);
    `secondary_radius` overrides the secondary's aperture radius (e.g. to vignette the field)."""
    D, f, f1, b = diameter, focal_length, primary_focal_length, back_focus
    m = f / f1
    if m <= 1:
        raise ValueError("a Cassegrain's focal length exceeds its primary's")
    beta = b / f1
    k = (1 + beta) / (1 + m)
    d = f1 * (1 - k)
    R1, R2 = -2 * f1, -2 * k * f1 * m / (m - 1)
    if kind == "cassegrain":
        K1, K2 = -1.0, -((m + 1) / (m - 1)) ** 2
    elif kind == "ritchey-chretien":
        K1 = -1 - 2 * (1 + beta) / (m * m * (m - beta))
        K2 = -((m + 1) / (m - 1)) ** 2 - 2 * m * (m + 1) / ((m - beta) * (m - 1) ** 3)
    else:
        raise ValueError(f"unknown two-mirror kind {kind}")

    theta = field_radius + abs(field_bias)
    # the secondary: the marginal rays' footprint (k times the primary's), and the field's (its chief ray meets it
    # d tan θ from the axis)
    r2 = secondary_radius if secondary_radius is not None else k * D / 2 + d * math.tan(theta) + margin
    y2 = k * off_axis
    # the converging beam crosses the primary's plane within b/(d+b) of the secondary's footprint, plus the field
    hole = 0.0 if off_axis else min(r2, r2 * b / (d + b) + f * math.tan(theta) + margin)
    if off_axis and off_axis - D / 2 < y2 + r2 + margin:
        raise ValueError("the off-axis segment is not clear of the secondary's shadow: move it further off the axis")

    elements = [
        Element("mirror", "primary", D / 2, translation(), radius=R1, conic=K1, hole_radius=hole,
                aperture_centre=(0.0, off_axis)),
        Element("mirror", "secondary", r2, translation(z=-d), radius=R2, conic=K2, aperture_centre=(0.0, y2)),
    ]
    # an object north (+y) of the line of sight images at y = -f tan θ
    focal_plane = translation(y=-f * math.tan(field_bias), z=b) @ UPRIGHT
    if fold is not None:
        # a flat mirror at 45°, `fold` before the focal plane, reflecting +z into +y
        zf = b - fold
        n = np.array([0, 1, -1]) / math.sqrt(2)
        beam = r2 * fold / (d + b) + f * math.tan(theta)
        elements.append(Element("mirror", "fold", math.sqrt(2) * beam + margin,
                                frame((0, 0, zf), (1, 0, 0), np.cross(n, (1, 0, 0)), n)))
        reflect = np.eye(4)
        reflect[:3, :3] -= 2 * np.outer(n, n)
        reflect[:3, 3] = 2 * zf * n[2] * n      # about the plane through (0, 0, zf)
        folded = reflect @ focal_plane
        folded[:3, 0] *= -1                     # a rotation again: the image is mirrored
        focal_plane = folded
    obscuration = 0.0 if off_axis else max(r2, hole) / (D / 2)
    return Design(f"{kind} f/{f / D:.3g}", elements, focal_plane, f, D, obscuration)


def hubble():
    """The Hubble Space Telescope (Laikin, Lens Design, 4th ed., p. 200): 2.4 m, f/24, with Laikin's central
    obscuration (177.8 mm radius, on the primary) and a 0.3 m secondary."""
    mm = 1e-3
    d, image = 4910.01016 * mm, 6365.20955 * mm
    elements = [
        Element("mirror", "primary", 1200 * mm, translation(), radius=-11040.02286 * mm, conic=-1.001152,
                hole_radius=177.80035 * mm),
        Element("mirror", "secondary", 150 * mm, translation(z=-d), radius=-1349.31166 * mm, conic=-1.483014),
    ]
    focal_plane = translation(z=image - d) @ UPRIGHT
    return Design("Hubble Space Telescope", elements, focal_plane, 57.6, 2.4, 177.80035 / 1200)


# <coating> elements of the telescope sensor
COATINGS = {
    "ideal": '<coating type="ideal"/>',
    "aluminium": '<coating type="multilayer"><spectrum name="substrate" material="Al"/></coating>',
    "silver": '<coating type="multilayer"><spectrum name="substrate" material="Ag"/></coating>',
    "gold": '<coating type="multilayer"><spectrum name="substrate" material="Au"/></coating>',
    "protected-aluminium": ('<coating type="multilayer"><spectrum name="substrate" material="Al"/>'
                            '<layer><spectrum name="IOR" constant="1.38"/>'
                            '<quantity name="thickness" value="25nm"/></layer></coating>'),
}


def coat(design, coating):
    """The design with every mirror coated with `coating` (a name of COATINGS)."""
    if coating not in COATINGS:
        raise ValueError(f"unknown coating {coating}: one of {', '.join(COATINGS)}")
    return dataclasses.replace(design, elements=[dataclasses.replace(e, coating=coating) if e.kind == "mirror" else e
                                                 for e in design.elements])


def _length(v):
    return f"{v:.12g}m"


def _transform(name, m, indent):
    rows = []
    for r in range(3):
        rows.append(", ".join(f"{m[r, c]:.17g}" for c in range(3)) + f", {_length(m[r, 3])}")
    rows.append("0, 0, 0, 1")
    return (f'{indent}<transform name="{name}">\n'
            f'{indent}    <matrix value="{", ".join(rows)}"/>\n'
            f'{indent}</transform>')


def elements_xml(design, indent="    "):
    """The design's elements and focal plane, as the telescope sensor's children."""
    out = []
    for e in design.elements:
        out.append(f'{indent}<{e.kind} id="{e.id}">')
        i2 = indent + "    "
        if e.radius is not None:
            out.append(f'{i2}<quantity name="radius" value="{_length(e.radius)}"/>')
        if e.conic:
            out.append(f'{i2}<float name="conic" value="{e.conic:.17g}"/>')
        for i, a in enumerate(e.aspheric):
            out.append(f'{i2}<float name="a{4 + 2 * i}" value="{a:.17g}"/>')
        out.append(f'{i2}<quantity name="aperture_radius" value="{_length(e.aperture_radius)}"/>')
        if e.hole_radius:
            out.append(f'{i2}<quantity name="hole_radius" value="{_length(e.hole_radius)}"/>')
        if any(e.aperture_centre):
            cx, cy = e.aperture_centre
            out.append(f'{i2}<vector name="aperture_centre" value="{_length(cx)}, {_length(cy)}"/>')
        out.append(_transform("to_telescope", e.to_telescope, i2))
        if e.kind == "mirror" and e.coating != "ideal":
            out.append(i2 + COATINGS[e.coating])
        out.append(f"{indent}</{e.kind}>")
    out.append(_transform("focal_plane", design.focal_plane, indent))
    return "\n".join(out)


def sensor_xml(design, pixel_pitch, pixels, placement, samples="$spp", band_nm=(540, 560), extra="", id="camera"):
    """A <sensor type="telescope"> of the design: `placement` maps the placement's attributes (altitude,
    view_zenith, ...) to their values, as strings; the film is square, of a band_radiance response of one box band."""
    width, height = pixels if isinstance(pixels, tuple) else (pixels, pixels)
    lines = [f'<sensor type="telescope" id="{id}">',
             f"    <!-- {design.name}: f = {design.focal_length:.6g} m, D = {design.entrance_pupil_diameter:.6g} m, "
             f"obscuration {design.obscuration:.3g} -->"]
    lines += [f'    <quantity name="{k}" value="{v}"/>' for k, v in placement.items()]
    lines += [f'    <quantity name="pixel_pitch" value="{_length(pixel_pitch)}"/>',
              f'    <quantity name="focal_length" value="{_length(design.focal_length)}"/>',
              f'    <integer name="samples" value="{samples}"/>']
    if extra:
        lines.append(extra)
    lines.append(elements_xml(design))
    lines += ['    <film type="array">',
              f'        <integer name="width" value="{width}"/>',
              f'        <integer name="height" value="{height}"/>',
              '        <response type="band_radiance">',
              '            <spectrum type="piecewise_linear">',
              f'                <bin wavelength="{band_nm[0]}nm" value="1"/>',
              f'                <bin wavelength="{band_nm[1]}nm" value="1"/>',
              '            </spectrum>',
              '        </response>',
              '    </film>',
              '</sensor>']
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("design", choices=["two-mirror", "hubble"])
    p.add_argument("--kind", choices=["cassegrain", "ritchey-chretien"], default="ritchey-chretien")
    p.add_argument("--diameter", type=float, default=0.4, help="m (default 0.4)")
    p.add_argument("--focal-length", type=float, default=2.5, help="m (default 2.5)")
    p.add_argument("--primary-focal-length", type=float, default=0.6, help="m (default 0.6)")
    p.add_argument("--back-focus", type=float, default=0.15, help="m, behind the primary (default 0.15)")
    p.add_argument("--off-axis", type=float, default=0.0, help="m: an off-axis segment, its centre this far off the axis")
    p.add_argument("--field-bias", type=float, default=0.0, help="degrees: the film's centre at this field angle")
    p.add_argument("--fold", type=float, help="m: a fold mirror this far before the focal plane")
    p.add_argument("--secondary-radius", type=float, help="m: the secondary's aperture radius (default: the field's)")
    p.add_argument("--coating", choices=list(COATINGS), default="ideal", help="every mirror's (default ideal)")
    p.add_argument("--pixel-pitch", type=float, default=5e-6, help="m (default 5 µm)")
    p.add_argument("--pixels", type=int, default=256, help="film width and height (default 256)")
    p.add_argument("--altitude", default="500km")
    p.add_argument("--view-zenith", default="0°")
    p.add_argument("--view-azimuth", default="0°")
    p.add_argument("--samples", default="$spp")
    p.add_argument("--elements-only", action="store_true",
                   help="print only the sensor's pixel pitch, focal length, elements and focal plane: to paste into a "
                        "<sensor type=\"telescope\"> of one's own")
    args = p.parse_args()

    if args.design == "hubble":
        design = hubble()
    else:
        # the secondary and the hole pass the film's corners
        field = math.atan(args.pixels * args.pixel_pitch / math.sqrt(2) / args.focal_length)
        design = two_mirror(args.kind, args.diameter, args.focal_length, args.primary_focal_length, args.back_focus,
                            field_radius=field, off_axis=args.off_axis, field_bias=math.radians(args.field_bias),
                            fold=args.fold, secondary_radius=args.secondary_radius)
    design = coat(design, args.coating)
    if args.elements_only:
        print(f"<!-- {design.name}: f = {design.focal_length:.6g} m, D = {design.entrance_pupil_diameter:.6g} m, "
              f"obscuration {design.obscuration:.3g}; for a {args.pixels}² film -->")
        print(f'<quantity name="pixel_pitch" value="{_length(args.pixel_pitch)}"/>')
        print(f'<quantity name="focal_length" value="{_length(design.focal_length)}"/>')
        print(elements_xml(design, indent=""))
        return
    placement = {"altitude": args.altitude, "view_zenith": args.view_zenith, "view_azimuth": args.view_azimuth}
    print(sensor_xml(design, args.pixel_pitch, args.pixels, placement, samples=args.samples))


if __name__ == "__main__":
    main()
