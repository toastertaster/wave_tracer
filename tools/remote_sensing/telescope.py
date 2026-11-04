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
  three-mirror an unobscured three-mirror anastigmat (as Cook's): an off-axis segment of a coaxial parent of a concave
               primary, a convex secondary and a concave tertiary behind the primary, with an intermediate image
               between the secondary and the tertiary. The parent's radii are solved for the focal length f and a
               flat field (zero Petzval sum, 1/R1 - 1/R2 + 1/R3 = 0) from the primary's focal length f1 and the
               mirrors' separations (--separations d1,d2: primary to secondary, secondary to tertiary); its conic
               constants, and the focal plane's distance, by tracing rays through the segment (diameter D, centred
               --off-axis Y from the axis) over the field (--field-bias off the axis, towards north) and making
               their images as small as they go (Levenberg-Marquardt: a third-order anastigmat, to the rays'
               precision). The secondary's and tertiary's apertures cover the traced beams' footprints.
  hubble       the Hubble Space Telescope's Ritchey-Chrétien (Laikin, Lens Design, 4th ed., p. 200, as in
               Optiland's sample), with a flat focal plane at its image surface's vertex.

--curved-focal-plane replaces the flat focal plane of a design of coaxial mirrors (unfolded) by the sphere, centred on
the axis, on which the film's field images the smallest (its vertex and radius by Levenberg-Marquardt, on rays traced
through the aperture): the film is tangent to it at the field that images at the flat focal plane's centre, tilted to
it off the axis (the sensor's focal_plane_radius).

The telescope's frame follows optical design software's convention: light arrives travelling towards +z, the primary
at the origin. The film's x is east and its y south at nadir (as a satellite_frame camera's) for designs with an even
number of mirrors. Lengths in the scene are in metres.

--coating coats every mirror (default ideal: a perfect conductor, reflecting all light):
  aluminium, silver, gold   the bare metals (wave_tracer's refractive-index data)
  protected-aluminium       aluminium under 25 nm of magnesium fluoride (n = 1.38), as Hubble's mirrors

Stray light (the sensor's stray_light and pst_angles): --scatter A,B,g makes every mirror rough, scattering by the
ABg model; --baffles adds, to an on-axis two-mirror design, the baffles that keep the focal plane from seeing the scene
past the secondary: a tube in front of the secondary (20% wider than it), a tube in front of the primary's hole (as
wide as the converging beam there; for a wide film, long enough that the film sees no further past the secondary's edge
than the secondary's tube), the secondary's tube long enough that the lines from the film past the secondary's edge
meet it, and a vane over the part of the primary in the secondary's shadow (where light would reflect between the
mirrors); --barrel encloses the telescope in a tube (from an aperture's diameter in front of the foremost mirror to
behind the focal plane), without which the focal plane sees the scene around the primary's rim; of
--baffle-reflectance (default 0, black).

    uv run tools/remote_sensing/telescope.py two-mirror --kind ritchey-chretien [--diameter 0.4 --focal-length 2.5 ...] > sensor.xml

prints the sensor (with --altitude, --view-zenith, ... placing it, as strings, so that scene defines like $vz work).
As a module, `two_mirror()`, `three_mirror()` and `hubble()` return designs, `coat()`, `roughen()`, `baffle()`,
`barrel()` and `curve_focal_plane()` modify them, `sensor_xml()` writes them, and `band_throughput()` gives the
fraction of a band that their mirrors' coatings reflect (at normal incidence; for toa.py's path radiance).
"""

import argparse
import dataclasses
import math
import pathlib

import numpy as np


@dataclasses.dataclass
class Element:
    """A mirror, stop, obscuration or tube of the telescope sensor: its shape, clear aperture and placement (4×4,
    element frame to telescope frame, metres)."""
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
    scatter: tuple[float, float, float] | None = None   # a mirror's ABg
    length: float = 0.0                                 # a tube's
    reflectance: float = 0.0                            # a stop's, obscuration's or tube's (Lambertian)


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
    # the image's scale about the film's centre (m/rad), as the sensor finds its focal lengths, if it is not the
    # focal length (a biased field's)
    film_centre_scale: float | None = None
    # a curved focal plane's radius of curvature (m; its centre on the focal plane frame's +z side if positive), or
    # None for a flat one
    focal_plane_radius: float | None = None


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


# ----------------------------------------------------------------------------------------------------------------
# three-mirror anastigmats: exact ray tracing of coaxial conic mirrors (numpy)

def _paraxial(curvatures, zs):
    """A marginal ray (height 1, parallel to the axis) through coaxial mirrors: its heights on them, and the
    system's focal length and paraxial image's z."""
    y, u, heights = 1.0, 0.0, []
    for j, c in enumerate(curvatures):
        if j:
            y += (zs[j] - zs[j - 1]) * u
        heights.append(y)
        u = -u - 2 * y * c
    return heights, -1.0 / u, zs[-1] - y / u


def _intersect(c, k, zv, o, d):
    """The rays (o, d) (N×3) with the conic of curvature c and conic constant k, vertex at z = zv: the nearest root
    beyond 0 on the branch through the vertex ((1+k) c z <= 1), NaN if none (as the telescope sensor)."""
    o = o - np.array([0.0, 0.0, zv])
    k1 = 1 + k
    A = c * (d[:, 0] ** 2 + d[:, 1] ** 2 + k1 * d[:, 2] ** 2)
    B = c * (o[:, 0] * d[:, 0] + o[:, 1] * d[:, 1] + k1 * o[:, 2] * d[:, 2]) - d[:, 2]
    C = c * (o[:, 0] ** 2 + o[:, 1] ** 2 + k1 * o[:, 2] ** 2) - 2 * o[:, 2]
    D = B * B - A * C
    with np.errstate(divide="ignore", invalid="ignore"):
        q = -(B + np.copysign(np.sqrt(np.maximum(D, 0)), B))
        r1 = np.where(A != 0, q / A, -C / (2 * B))
        r2 = np.where(q != 0, C / q, r1)
    t = np.full(len(o), np.nan)
    for roots in (np.minimum(r1, r2), np.maximum(r1, r2)):
        ok = np.isnan(t) & (D >= 0) & (roots > 1e-9) & (k1 * c * (o[:, 2] + roots * d[:, 2]) <= 1 + 1e-12)
        t = np.where(ok, roots, t)
    return t


def _trace(curvatures, conics, zs, field, pupil, z_image=None, c_image=0.0):
    """Rays from the field direction `field` = (θx, θy) (radians; light travelling +z) through the points `pupil`
    (M×2) of the plane z = 0, reflected by the mirrors in turn: their points on each mirror, and on the image surface
    (if `z_image` is given): the plane z = z_image, or the sphere of curvature `c_image` whose vertex is there."""
    d = np.array([math.tan(field[0]), math.tan(field[1]), 1.0])
    d /= np.linalg.norm(d)
    o = np.zeros((len(pupil), 3))
    o[:, :2] = pupil
    o -= 10 * d
    dd = np.tile(d, (len(pupil), 1))
    points = []
    for c, k, z in zip(curvatures, conics, zs):
        t = _intersect(c, k, z, o, dd)
        p = o + t[:, None] * dd
        r2 = p[:, 0] ** 2 + p[:, 1] ** 2
        g = c / np.sqrt(1 - (1 + k) * c * c * r2)
        n = np.stack([-p[:, 0] * g, -p[:, 1] * g, np.ones(len(p))], axis=1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        dd = dd - 2 * np.sum(dd * n, axis=1, keepdims=True) * n
        o = p
        points.append(p)
    if z_image is not None:
        t = _intersect(c_image, 0.0, z_image, o, dd) if c_image else (z_image - o[:, 2]) / dd[:, 2]
        points.append(o + t[:, None] * dd)
    return points


def _disc(centre_y, radius, n=11):
    g = np.linspace(-1, 1, n)
    x, y = np.meshgrid(g, g)
    inside = x * x + y * y <= 1
    return np.stack([radius * x[inside], centre_y + radius * y[inside]], axis=1)


def three_mirror(diameter=0.2, focal_length=2.5, primary_focal_length=0.6, separations=(0.45, 1.15), off_axis=0.2,
                 field_bias=math.radians(1.0), field_radius=math.radians(0.2), margin=1e-3):
    """An unobscured three-mirror anastigmat (see the module's doc). Angles in radians, lengths in metres; the
    conic constants are made anastigmatic over `field_radius` about `field_bias` (no less than 0.2°, so that they
    are well determined), the secondary's and tertiary's apertures cover the beams of `field_radius`."""
    f, f1, (d1, d2), D, Y = focal_length, primary_focal_length, separations, diameter, off_axis
    c1 = -1 / (2 * f1)
    zs = [0.0, -d1, -d1 + d2]

    # the secondary's curvature (and the tertiary's, for a flat field: c1 - c2 + c3 = 0) for the focal length: the
    # root whose marginal ray crosses the axis between the secondary and the tertiary (an intermediate image), and
    # whose image is between them
    def residual(c2):
        heights, efl, z_image = _paraxial([c1, c2, c2 - c1], zs)
        return abs(efl) - f
    grid = np.linspace(-1 / (0.02 * f1), c1, 20001)[:-1]
    values = np.array([residual(c) for c in grid])
    c2 = None
    for i in np.nonzero(np.sign(values[:-1]) != np.sign(values[1:]))[0]:
        a, b = grid[i], grid[i + 1]
        for _ in range(80):
            m = (a + b) / 2
            if np.sign(residual(m)) == np.sign(residual(a)):
                a = m
            else:
                b = m
        heights, efl, z_image = _paraxial([c1, a, a - c1], zs)
        if heights[2] < 0 and zs[1] < z_image < zs[2]:
            c2 = a
            break
    if c2 is None:
        raise ValueError("no three-mirror anastigmat of this focal length with an intermediate image between the "
                         "secondary and the tertiary: change the separations")
    curvatures = [c1, c2, c2 - c1]
    heights, efl, z_image = _paraxial(curvatures, zs)

    # conic constants and focus: the smallest images, over the segment and field (Levenberg-Marquardt)
    pupil = _disc(Y, D / 2)
    df = max(field_radius, math.radians(0.2))
    fields = [(0, field_bias), (df, field_bias), (-df, field_bias), (0, field_bias + df), (0, field_bias - df)]

    def residuals(x):
        out = []
        for field in fields:
            p = _trace(curvatures, x[:3], zs, field, pupil, z_image=x[3])[-1][:, :2]
            out.append((p - p.mean(axis=0)).ravel())
        return np.concatenate(out)

    x = np.array([-1.0, -2.0, -0.5, z_image])
    r = residuals(x)
    lam = 1e-3
    for _ in range(100):
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            h = 1e-6 * max(1.0, abs(x[j]))
            xp = x.copy()
            xp[j] += h
            J[:, j] = (residuals(xp) - r) / h
        improved = False
        while lam < 1e12:
            H = J.T @ J
            step = np.linalg.solve(H + lam * np.diag(np.diag(H)), -J.T @ r)
            rn = residuals(x + step)
            if np.all(np.isfinite(rn)) and rn @ rn < r @ r:
                x, r, lam, improved = x + step, rn, lam * 0.3, True
                break
            lam *= 10
        if not improved or np.max(np.abs(step)) < 1e-12:
            break
    conics, z_image = list(x[:3]), float(x[3])

    # the secondary's and tertiary's apertures: the footprints of the field's beams (its corners included)
    footprint = [[], [], []]
    image_centre = None
    for fx in np.linspace(-field_radius, field_radius, 5):
        for fy in np.linspace(field_bias - field_radius, field_bias + field_radius, 5):
            points = _trace(curvatures, conics, zs, (fx, fy), _disc(Y, D / 2, 21), z_image=z_image)
            for j in range(3):
                footprint[j].append(points[j][:, :2])
            if fx == 0 and abs(fy - field_bias) < 1e-15:
                image_centre = points[3][:, :2].mean(axis=0)
    apertures = []
    for j in (1, 2):
        p = np.concatenate(footprint[j])
        centre = np.array([0.0, (p[:, 1].min() + p[:, 1].max()) / 2])
        apertures.append((float(np.max(np.linalg.norm(p - centre, axis=1))) + margin, (0.0, float(centre[1]))))

    elements = [
        Element("mirror", "primary", D / 2, translation(), radius=1 / c1, conic=conics[0], aperture_centre=(0.0, Y)),
        Element("mirror", "secondary", apertures[0][0], translation(z=zs[1]), radius=1 / c2, conic=conics[1],
                aperture_centre=apertures[0][1]),
        Element("mirror", "tertiary", apertures[1][0], translation(z=zs[2]), radius=1 / curvatures[2],
                conic=conics[2], aperture_centre=apertures[1][1]),
    ]
    # the focal plane faces the tertiary (light arrives travelling -z), centred on the biased field's image; with
    # three mirrors, the image is mirrored
    focal_plane = frame((image_centre[0], image_centre[1], z_image), (1, 0, 0), (0, 1, 0), (0, 0, 1))

    # the image's scale about the biased field: its centroid's motion per angle, along x and y (the sensor checks its
    # `focal_length` against their mean)
    def centroid(fx, fy):
        return _trace(curvatures, conics, zs, (fx, fy), pupil, z_image=z_image)[-1][:, :2].mean(axis=0)

    def direction(fx, fy):
        v = np.array([math.tan(fx), math.tan(fy), 1.0])
        return v / np.linalg.norm(v)
    delta = 1e-5
    scale_x = (np.linalg.norm(centroid(delta, field_bias) - centroid(-delta, field_bias)) /
               np.linalg.norm(direction(delta, field_bias) - direction(-delta, field_bias)))
    scale_y = (np.linalg.norm(centroid(0, field_bias + delta) - centroid(0, field_bias - delta)) /
               np.linalg.norm(direction(0, field_bias + delta) - direction(0, field_bias - delta)))
    return Design(f"three-mirror anastigmat f/{f / D:.3g}", elements, focal_plane, f, D, 0.0,
                  film_centre_scale=float((scale_x + scale_y) / 2))


def curve_focal_plane(design, field_radius):
    """The design (of coaxial mirrors) with a curved focal plane: the sphere, centred on the axis, on which the
    images of the field (`field_radius` radians about the field that images at the flat focal plane's centre) are the
    smallest (its vertex and curvature by Levenberg-Marquardt), the film tangent to it at that field's image."""
    mirrors = design.elements
    for e in mirrors:
        if e.kind != "mirror" or not np.allclose(e.to_telescope[:3, :3], np.eye(3)) or any(e.to_telescope[:2, 3]):
            raise ValueError("a curved focal plane is fitted to designs of coaxial mirrors only (no fold, baffles)")
    curvatures = [1 / e.radius if e.radius else 0.0 for e in mirrors]
    conics = [e.conic for e in mirrors]
    zs = [float(e.to_telescope[2, 3]) for e in mirrors]
    primary = mirrors[0]
    pupil = _disc(primary.aperture_centre[1], primary.aperture_radius, 21)
    # (not the central obscuration's)
    pupil = pupil[np.hypot(pupil[:, 0], pupil[:, 1]) >= design.obscuration * primary.aperture_radius]

    # the field (along y: `_trace`'s direction of travel) that images at the flat focal plane's centre (Newton)
    def centre_y(fy):
        return _trace(curvatures, conics, zs, (0.0, fy), pupil, z_image=design.focal_plane[2, 3])[-1][:, 1].mean()
    field_bias = 0.0
    for _ in range(8):
        h = 1e-6
        slope = (centre_y(field_bias + h) - centre_y(field_bias - h)) / (2 * h)
        field_bias -= (centre_y(field_bias) - design.focal_plane[1, 3]) / slope
    fields = [(fx, fy) for fx in np.linspace(-field_radius, field_radius, 5)
              for fy in np.linspace(field_bias - field_radius, field_bias + field_radius, 5)]
    # the focal plane's frame's z faces the optics: telescope z along it is `side` times local z
    side = float(np.sign(design.focal_plane[2, 2]))

    def images(x, field, rays=pupil):
        return _trace(curvatures, conics, zs, field, rays, z_image=x[0], c_image=x[1])[-1]

    def residuals(x):
        out = []
        for field in fields:
            p = images(x, field)[:, :2]
            out.append((p - p.mean(axis=0)).ravel())
        return np.concatenate(out)

    # (vertex z, curvature), from the flat focal plane's
    x = np.array([float(design.focal_plane[2, 3]), 0.0])
    r = residuals(x)
    lam = 1e-3
    for _ in range(100):
        J = np.empty((len(r), len(x)))
        for j, h in enumerate((1e-7, 1e-6)):
            xp = x.copy()
            xp[j] += h
            J[:, j] = (residuals(xp) - r) / h
        improved = False
        while lam < 1e12:
            H = J.T @ J
            step = np.linalg.solve(H + lam * np.diag(np.diag(H)), -J.T @ r)
            rn = residuals(x + step)
            if np.all(np.isfinite(rn)) and rn @ rn < r @ r:
                x, r, lam, improved = x + step, rn, lam * 0.3, True
                break
            lam *= 10
        if not improved or np.max(np.abs(step)) < 1e-14:
            break
    z_vertex, c = float(x[0]), float(x[1])

    # the film's centre: the biased field's image, on the sphere; its normal there (facing the optics, as the flat
    # focal plane's z); its x as the flat focal plane's
    p = images(x, (0.0, field_bias)).mean(axis=0)
    # (back on the sphere: the centroid of points on it is not)
    r2 = p[0] ** 2 + p[1] ** 2
    p[2] = z_vertex + (c * r2 / (1 + math.sqrt(1 - c * c * r2)) if c else 0.0)
    n = np.array([-p[0] * c, -p[1] * c, 1 - c * (p[2] - z_vertex)])
    z_axis = side * n / np.linalg.norm(n)
    x_axis = design.focal_plane[:3, 0] - np.dot(design.focal_plane[:3, 0], z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    focal_plane = frame(p, x_axis, y_axis, z_axis)

    # the image's scale about the biased field, on the sphere (as `three_mirror`)
    def centroid(fx, fy):
        return images(x, (fx, fy)).mean(axis=0)

    def direction(fx, fy):
        v = np.array([math.tan(fx), math.tan(fy), 1.0])
        return v / np.linalg.norm(v)
    delta = 1e-5
    scale_x = (np.linalg.norm(centroid(delta, field_bias) - centroid(-delta, field_bias)) /
               np.linalg.norm(direction(delta, field_bias) - direction(-delta, field_bias)))
    scale_y = (np.linalg.norm(centroid(0, field_bias + delta) - centroid(0, field_bias - delta)) /
               np.linalg.norm(direction(0, field_bias + delta) - direction(0, field_bias - delta)))
    return dataclasses.replace(design, focal_plane=focal_plane, film_centre_scale=float((scale_x + scale_y) / 2),
                               focal_plane_radius=side / c if c else None,
                               name=design.name + ", curved focal plane")


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


# the coatings: a substrate (a material of wave_tracer's data/ior) under layers (refractive index, thickness in
# nm; outermost first), or None for the ideal coating
COATING_STACKS = {
    "ideal": None,
    "aluminium": ("Al", []),
    "silver": ("Ag", []),
    "gold": ("Au", []),
    "protected-aluminium": ("Al", [(1.38, 25)]),
}


def _coating_xml(stack):
    if stack is None:
        return '<coating type="ideal"/>'
    substrate, layers = stack
    return (f'<coating type="multilayer"><spectrum name="substrate" material="{substrate}"/>'
            + "".join(f'<layer><spectrum name="IOR" constant="{n:g}"/><quantity name="thickness" value="{d:g}nm"/>'
                      f'</layer>' for n, d in layers) + "</coating>")


# <coating> elements of the telescope sensor
COATINGS = {name: _coating_xml(stack) for name, stack in COATING_STACKS.items()}

IOR_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "ior"


def _tabulated_nk(material):
    """A refractiveindex.info file's tabulated n, k: wavelengths (nm), n + ik."""
    text = (IOR_DIR / f"{material}.yml").read_text()
    rows = text.split("type: tabulated nk", 1)[1].split("data: |", 1)[1]
    data = []
    for line in rows.splitlines()[1:]:
        parts = line.split()
        if len(parts) != 3:
            break
        data.append([float(p) for p in parts])
    data = np.array(data)
    return data[:, 0] * 1e3, data[:, 1] + 1j * data[:, 2]


def coating_reflectance(coating, wavelengths_nm):
    """The coating's reflectance (unpolarized) at normal incidence, at each wavelength: the layers'
    characteristic matrices (Macleod, Thin-Film Optical Filters, ch. 2), as the sensor computes it."""
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    stack = COATING_STACKS[coating]
    if stack is None:
        return np.ones_like(wavelengths)
    substrate, layers = stack
    table_nm, table_nk = _tabulated_nk(substrate)
    nk = np.interp(wavelengths, table_nm, table_nk.real) + 1j * np.interp(wavelengths, table_nm, table_nk.imag)
    # Macleod's admittances (N = n - ik, in free space's), from the substrate outwards
    b, c = np.ones_like(nk), np.conj(nk)
    for n, d in reversed(layers):
        delta = 2 * np.pi * n * d / wavelengths
        b, c = np.cos(delta) * b + 1j * np.sin(delta) / n * c, 1j * n * np.sin(delta) * b + np.cos(delta) * c
    return np.abs((b - c) / (b + c)) ** 2


def band_throughput(design, band_nm):
    """The fraction of light that the design's mirrors reflect, at normal incidence, averaged over the band
    (lo, hi nm; uniform in wavelength): the product of the mirrors' reflectances."""
    wavelengths = np.linspace(band_nm[0], band_nm[1], 64)
    t = np.ones_like(wavelengths)
    for e in design.elements:
        if e.kind == "mirror":
            t = t * coating_reflectance(e.coating, wavelengths)
    return float(t.mean())


def coat(design, coating):
    """The design with every mirror coated with `coating` (a name of COATINGS)."""
    if coating not in COATINGS:
        raise ValueError(f"unknown coating {coating}: one of {', '.join(COATINGS)}")
    return dataclasses.replace(design, elements=[dataclasses.replace(e, coating=coating) if e.kind == "mirror" else e
                                                 for e in design.elements])


def roughen(design, abg):
    """The design with every mirror scattering by the ABg model, `abg` = (A, B, g)."""
    return dataclasses.replace(design, elements=[dataclasses.replace(e, scatter=tuple(abg)) if e.kind == "mirror" else e
                                                 for e in design.elements])


def baffle(design, reflectance=0.0, margin=1e-3):
    """An on-axis two-mirror design with baffles (see the module's doc): a tube in front of the secondary, a tube in
    front of the primary's hole (if the film is wide enough to need it), and a vane over the primary's shadowed
    zone."""
    primary, secondary = design.elements[0], design.elements[1]
    if any(primary.aperture_centre) or primary.kind != "mirror" or secondary.kind != "mirror":
        raise ValueError("baffles are for on-axis two-mirror designs")
    r_s, r_h = secondary.aperture_radius, primary.hole_radius
    z_s = secondary.to_telescope[2, 3]
    b = design.focal_plane[2, 3] if design.focal_plane[2, 2] < 0 else None
    if b is None:
        raise ValueError("baffles are for designs whose focal plane is behind the primary, facing it")
    h_s = b - z_s           # the secondary, from the focal plane
    # the film's half-diagonal a, from the hole that passes its beams (two_mirror: r_s b/h_s + a + margin)
    a = max(0.0, r_h - r_s * b / h_s - margin)

    # the secondary's tube, of radius r_t = 1.2 r_s. The film sees past the secondary through the primary's hole: the
    # steepest such line, from the film's edge (-a) through the far side of an opening of radius r at a height h
    # above the focal plane, has the slope (r + a)/h, and crosses the secondary's plane (h_s) at (r + a) h_s/h - a.
    # Through the hole alone (r_h at b) that may be beyond the secondary's tube: then a tube in front of the hole, of
    # the converging beam's radius at its height, r_p(h) = r_s h/h_s + a + margin (it passes the beam), reaching h_p,
    # brings it within r_t if h_p >= (2a + margin) h_s/(r_t - r_s + a)
    r_t = 1.2 * r_s
    h_p = (2 * a + margin) * h_s / (r_t - r_s + a)
    elements = list(design.elements)
    if h_p > b + margin:
        r_p = r_s * h_p / h_s + a + margin
        steepest = (r_p + a) / h_p
        elements.append(Element("tube", "primary baffle", r_p, translation(z=b - h_p), length=h_p - b,
                                reflectance=reflectance))
    else:
        steepest = (r_h + a) / b
    # long enough that the steepest line meets it: it reaches r_t at h = (r_t + a)/steepest (and, without a primary
    # tube, at least as long as the centre's view past the edge needs, (b - z_s)(1.2 - 1) 1.5)
    length = max(h_s * (r_t / r_s - 1) * 1.5, (r_t + a) / steepest - h_s + margin)
    elements.append(Element("tube", "secondary baffle", r_t, translation(z=z_s - length), length=length,
                            reflectance=reflectance))
    if r_h + margin < r_s - margin:
        elements.append(Element("obscuration", "vane", r_s - margin, translation(z=-2 * margin),
                                hole_radius=r_h + margin, reflectance=reflectance))
    return dataclasses.replace(design, elements=elements)


def barrel(design, reflectance=0.0, field_radius=math.radians(0.2), margin=0.01):
    """The design (of mirrors about the telescope's z axis, light arriving towards +z) inside a barrel: a tube about
    the axis enclosing the mirrors' apertures (and the field's beams, `field_radius` off the axis, plus `margin`), from
    an aperture's diameter in front of the foremost mirror to behind the focal plane. Without it, the focal plane sees
    the scene around the primary's rim."""
    mirrors = [e for e in design.elements if e.kind == "mirror"]
    if any(not np.allclose(e.to_telescope[:3, :3], np.eye(3)) or any(e.to_telescope[:2, 3]) for e in mirrors):
        raise ValueError("a barrel is for designs of mirrors about the telescope's axis (no fold)")
    D = 2 * mirrors[0].aperture_radius
    zs = [float(e.to_telescope[2, 3]) for e in mirrors] + [float(design.focal_plane[2, 3])]
    front, back = min(zs) - D, max(zs) + 0.05 * D
    extent = max(math.hypot(*e.aperture_centre) + e.aperture_radius for e in mirrors)
    radius = extent + (back - front) * math.tan(field_radius) + margin
    elements = list(design.elements)
    elements.append(Element("tube", "barrel", radius, translation(z=front), length=back - front,
                            reflectance=reflectance))
    return dataclasses.replace(design, elements=elements)


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
        if e.kind == "tube":
            out.append(f'{i2}<quantity name="length" value="{_length(e.length)}"/>')
        if e.reflectance:
            out.append(f'{i2}<float name="reflectance" value="{e.reflectance:.6g}"/>')
        if e.kind == "mirror" and e.coating != "ideal":
            out.append(i2 + COATINGS[e.coating])
        if e.kind == "mirror" and e.scatter:
            a, b, g = e.scatter
            out.append(f'{i2}<scatter type="abg"><float name="A" value="{a:.9g}"/><float name="B" value="{b:.9g}"/>'
                       f'<float name="g" value="{g:.9g}"/></scatter>')
        out.append(f"{indent}</{e.kind}>")
    out.append(_transform("focal_plane", design.focal_plane, indent))
    if design.focal_plane_radius:
        out.append(f'{indent}<quantity name="focal_plane_radius" value="{_length(design.focal_plane_radius)}"/>')
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
              f'    <quantity name="focal_length" value="{_length(design.film_centre_scale or design.focal_length)}"/>',
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
    p.add_argument("design", choices=["two-mirror", "three-mirror", "hubble"])
    p.add_argument("--kind", choices=["cassegrain", "ritchey-chretien"], default="ritchey-chretien")
    p.add_argument("--diameter", type=float, default=0.4, help="m (default 0.4)")
    p.add_argument("--focal-length", type=float, default=2.5, help="m (default 2.5)")
    p.add_argument("--primary-focal-length", type=float, default=0.6, help="m (default 0.6)")
    p.add_argument("--back-focus", type=float, default=0.15, help="m, behind the primary (default 0.15)")
    p.add_argument("--off-axis", type=float, default=0.0, help="m: an off-axis segment, its centre this far off the axis")
    p.add_argument("--field-bias", type=float, default=0.0, help="degrees: the film's centre at this field angle")
    p.add_argument("--fold", type=float, help="m: a fold mirror this far before the focal plane")
    p.add_argument("--separations", default="0.45,1.15",
                   help="three-mirror: m, primary to secondary and secondary to tertiary (default 0.45,1.15)")
    p.add_argument("--secondary-radius", type=float, help="m: the secondary's aperture radius (default: the field's)")
    p.add_argument("--curved-focal-plane", action="store_true",
                   help="the sphere on which the film's field images best (coaxial mirrors only)")
    p.add_argument("--coating", choices=list(COATINGS), default="ideal", help="every mirror's (default ideal)")
    p.add_argument("--scatter", help="A,B,g: every mirror's ABg scatter (default: smooth)")
    p.add_argument("--baffles", action="store_true", help="baffle tubes before the secondary and the primary's hole, and a vane (on-axis two-mirror)")
    p.add_argument("--barrel", action="store_true", help="a tube enclosing the telescope (coaxial mirrors, unfolded)")
    p.add_argument("--baffle-reflectance", type=float, default=0.0,
                   help="the baffles' and barrel's (Lambertian; default 0)")
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
    elif args.design == "three-mirror":
        field = math.atan(args.pixels * args.pixel_pitch / math.sqrt(2) / args.focal_length)
        design = three_mirror(args.diameter, args.focal_length, args.primary_focal_length,
                              tuple(float(v) for v in args.separations.split(",")),
                              off_axis=args.off_axis or 0.2, field_bias=math.radians(args.field_bias or 1.0),
                              field_radius=field)
    else:
        # the secondary and the hole pass the film's corners
        field = math.atan(args.pixels * args.pixel_pitch / math.sqrt(2) / args.focal_length)
        design = two_mirror(args.kind, args.diameter, args.focal_length, args.primary_focal_length, args.back_focus,
                            field_radius=field, off_axis=args.off_axis, field_bias=math.radians(args.field_bias),
                            fold=args.fold, secondary_radius=args.secondary_radius)
    if args.curved_focal_plane:
        design = curve_focal_plane(design, math.atan(args.pixels * args.pixel_pitch / math.sqrt(2) / design.focal_length))
    design = coat(design, args.coating)
    if args.scatter:
        design = roughen(design, [float(v) for v in args.scatter.split(",")])
    if args.baffles:
        design = baffle(design, args.baffle_reflectance)
    if args.barrel:
        field = math.atan(args.pixels * args.pixel_pitch / math.sqrt(2) / design.focal_length)
        design = barrel(design, args.baffle_reflectance, field_radius=field + math.radians(args.field_bias))
    if args.elements_only:
        print(f"<!-- {design.name}: f = {design.focal_length:.6g} m, D = {design.entrance_pupil_diameter:.6g} m, "
              f"obscuration {design.obscuration:.3g}; for a {args.pixels}² film -->")
        print(f'<quantity name="pixel_pitch" value="{_length(args.pixel_pitch)}"/>')
        print(f'<quantity name="focal_length" value="{_length(design.film_centre_scale or design.focal_length)}"/>')
        print(elements_xml(design, indent=""))
        return
    placement = {"altitude": args.altitude, "view_zenith": args.view_zenith, "view_azimuth": args.view_azimuth}
    print(sensor_xml(design, args.pixel_pitch, args.pixels, placement, samples=args.samples))


if __name__ == "__main__":
    main()
