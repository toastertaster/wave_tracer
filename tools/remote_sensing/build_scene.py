# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "rasterio", "pillow"]
# ///
"""Builds a wave_tracer scene of a site fetched by fetch_site.py.

Writes into the site directory:
  terrain.ply      the terrain as a triangle mesh, with texture coordinates:
                   the lidar surface model (fetch_site.py --lidar: roofs, tree
                   crowns) at 1 m, with flat triangles; else (or with
                   --surface dem) the bare-earth DEM, with vertex normals
  terrain_<class>.ply
                   instead, with materials of classes (materials.py
                   --classes): the terrain's triangles of each class (by its
                   centroid in classes.png), the steep ones that are not
                   vegetation (walls) in terrain_wall.ply
  reflectance.png  without materials.json: a 16-bit linear reflectance map
  sky.ply          with an atmosphere: the sky, a hemisphere around the scene
                   (sky_<name>.ply, larger, with --surroundings)
  <name>.xml       the scene: the terrain, the sun, the sky, and a
                   satellite_frame camera recording band radiance (needs a
                   wave_tracer built with REALISTIC_REMOTE_SENSING); with
                   --telescope, a telescope camera instead, which traces its
                   rays through the mirrors of telescope.py's two-mirror
                   design (Cassegrain or Ritchey-Chretien) of the camera's
                   focal length, --telescope-aperture, --primary-f-number,
                   --back-focus and mirrors' --coating, or its three-mirror
                   anastigmat (its layout scaled to the focal length); an
                   off-axis segment (--telescope-off-axis) with its field off
                   the axis (--field-bias), on a --curved-focal-plane, with
                   --baffles and a --barrel; traced non-sequentially with
                   --stray-light, which sees the ground beyond the terrain
                   with --surroundings
  <name>.json      what toa.py and detector.py need: the bands, the camera
                   (and its telescope's aperture and mirrors' reflectance per
                   band, with --telescope), and the atmosphere's own radiance

The scene's frame is local to the site: x east, y up, z south (north is -z),
in metres, with the origin on the ground at the site's centre. The camera is
placed by altitude and view angles over that origin (see the satellite_frame
sensor). The sun shines from its position at the given time, with the TSIS-1
top-of-atmosphere spectrum scaled to the Earth-Sun distance of that day.

With the atmosphere that atmosphere.py computed (SITE_DIR/atmosphere.json, for
the same time and view; --no-atmosphere ignores it), per wavelength:
  - the sun's irradiance is attenuated by the direct transmittance and the
    gases' downward transmittance;
  - the sky is a hemisphere of radius --sky-radius around the scene, an area
    emitter of uniform radiance that gives a horizontal surface the diffuse
    irradiance 6S computes, including the light that the ground (of the
    scene's mean reflectance, beyond the scene too) and the atmosphere reflect
    back and forth. It is black, and has a hole around the sun through which
    the sun lights the whole scene; the camera's rays start inside it;
  - the camera sees the ground through the direct upward transmittance and
    the rest of the gases' two-way transmittance (the band_radiance response's
    transmittance).
The render is then the radiance at the top of the atmosphere that comes
straight from the ground. toa.py adds the rest: the atmosphere's path
radiance, and the ground's light that the atmosphere scatters into the view
(the adjacency effect).

The terrain's reflectance is that of the materials that materials.py mapped
(SITE_DIR/materials.json): a spectral_mix of their spectra, weighted by their
abundances (materials.exr); with classes, each class's own (materials_<class>
.exr), and the walls' one material. Without them, or with --reflectance, a
placeholder: spectrally flat, scaled from NAIP's brightness so that its median
is --reflectance (default .35). (NAIP is not radiometrically calibrated; its
RGB would also leave the near infrared dark.)

    uv run tools/remote_sensing/build_scene.py SITE_DIR [--gsd 1 --pixels 1920 --altitude 500e3 ...]

Render it with
    wave_tracer render [--gpu] SITE_DIR/scene.xml -o OUT_DIR
and, with an atmosphere, add its radiance with
    uv run tools/remote_sensing/toa.py SITE_DIR/scene.json OUT_DIR/camera.exr TOA.exr
"""

import argparse
import datetime
import json
import math
import pathlib

import numpy as np
import rasterio
from PIL import Image

import telescope
from sun import earth_sun_distance, solar_position

# placeholder multispectral bands (nm), as boxes: those of typical very-high-resolution imagers
BANDS = {"blue": (450, 520), "green": (520, 600), "red": (630, 690), "nir": (760, 900)}


def write_ply(path, vertices, normals, uvs, faces):
    """A binary PLY mesh; without vertex normals if `normals` is None."""
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(vertices)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 + ("" if normals is None else "property float nx\nproperty float ny\nproperty float nz\n") +
                 "property float u\nproperty float v\n"
                 f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        v = np.concatenate([vertices] + ([] if normals is None else [normals]) + [uvs], axis=1).astype("<f4")
        f.write(v.tobytes())
        fc = np.empty(len(faces), dtype=[("n", "u1"), ("i", "<i4", (3,))])
        fc["n"] = 3
        fc["i"] = faces
        f.write(fc.tobytes())


def terrain(dem_path, spacing: int, centre_en, centre_h):
    """A height map (the DEM, or the surface model), sampled every `spacing` pixels (at pixel centres), as a mesh in
    the scene frame."""
    with rasterio.open(dem_path) as ds:
        dem = ds.read(1).astype(np.float64)
        west, north = ds.transform.c, ds.transform.f
        res = ds.transform.a
    h, w = dem.shape
    rows = np.arange(0, h, spacing)
    cols = np.arange(0, w, spacing)
    east = west + (cols + 0.5) * res
    northing = north - (rows + 0.5) * res
    z = dem[np.ix_(rows, cols)]
    # normals from central differences of the full-resolution DEM
    dzdx = np.gradient(dem, res, axis=1)[np.ix_(rows, cols)]       # per metre east
    dzdy = -np.gradient(dem, res, axis=0)[np.ix_(rows, cols)]      # per metre north
    E, N = np.meshgrid(east, northing)
    vertices = np.stack([E - centre_en[0], z - centre_h, -(N - centre_en[1])], axis=-1).reshape(-1, 3)
    n = np.stack([-dzdx, np.ones_like(dzdx), dzdy], axis=-1)       # (x east, y up, z south)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    normals = n.reshape(-1, 3)
    # quads (r,c),(r,c+1),(r+1,c),(r+1,c+1); counter-clockwise seen from above
    nr, nc = len(rows), len(cols)
    idx = np.arange(nr * nc).reshape(nr, nc)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    faces = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])
    return vertices, normals, E.reshape(-1), N.reshape(-1), faces


def face_classes(vertices, E, N, faces, materials, site_dir, tex_bounds, wall_slope=60.0):
    """Each face's class of materials.json (an index into its classes; the wall after them): the class (classes.png)
    at its centroid; the wall if steeper than `wall_slope` degrees and not vegetation."""
    names = [c["name"] for c in materials["classes"]]
    ids = np.array([c["id"] for c in materials["classes"]])
    cmap = np.asarray(Image.open(site_dir / materials["class_map"]))
    h, w = cmap.shape
    ce, cn = E[faces].mean(axis=1), N[faces].mean(axis=1)
    col = np.clip(((ce - tex_bounds.left) / (tex_bounds.right - tex_bounds.left) * w).astype(int), 0, w - 1)
    row = np.clip(((tex_bounds.top - cn) / (tex_bounds.top - tex_bounds.bottom) * h).astype(int), 0, h - 1)
    lookup = np.full(256, names.index("ground") if "ground" in names else 0)     # no data (0): ground
    lookup[ids] = np.arange(len(ids))
    fc = lookup[cmap[row, col]]
    if materials.get("wall"):
        v = vertices[faces]
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        steep = np.abs(n[:, 1]) < np.cos(np.radians(wall_slope)) * np.linalg.norm(n, axis=1)
        vegetation = np.isin(fc, [names.index(c) for c in ("grass", "tree") if c in names])
        fc[steep & ~vegetation] = len(names)
    return fc


def write_class_meshes(site_dir, vertices, uvs, faces, fc, names):
    """A mesh of each class's faces, terrain_<name>.ply (without normals). Returns the classes with faces, and their
    numbers of faces."""
    written = []
    for i, name in enumerate(names):
        f = faces[fc == i]
        if not len(f):
            continue
        used, inverse = np.unique(f, return_inverse=True)
        write_ply(site_dir / f"terrain_{name}.ply", vertices[used], None, uvs[used], inverse.reshape(-1, 3))
        written.append((name, len(f)))
    return written


def reflectance_map(naip_path, out_path, mean_reflectance):
    """The placeholder reflectance: NAIP's brightness (mean of red, green, blue), scaled so that its median is
    `mean_reflectance`. Returns its mean."""
    with rasterio.open(naip_path) as ds:
        rgb = ds.read([1, 2, 3]).astype(np.float64)
    lum = rgb.mean(axis=0)
    rho = np.clip(lum / np.median(lum[lum > 0]) * mean_reflectance, 0, 1)
    Image.fromarray(np.round(rho * 65535).astype(np.uint16)).save(out_path)
    return rho.mean()


def terrain_bsdf(site_dir, nm, placeholder_reflectance, naip_path):
    """The terrain's diffuse BSDF (XML), and its mean reflectance at wavelengths `nm`: of the materials of
    materials.json, if any (and not `placeholder_reflectance`, a median reflectance), else the placeholder's."""
    materials_path = site_dir / "materials.json"
    if placeholder_reflectance is None and materials_path.exists():
        materials = json.loads(materials_path.read_text())
        assert "classes" not in materials       # see class_bsdfs
        wl = np.asarray(materials["wavelength_nm"])
        if nm[0] < wl[0] or nm[-1] > wl[-1]:
            raise SystemExit(f"{materials_path} covers {wl[0]:g}-{wl[-1]:g} nm, not {nm[0]:g}-{nm[-1]:g} nm")
        spectra = {name: np.interp(nm, wl, r) for name, r in materials["reflectance"].items()}
        mean = sum(materials["mean_abundances"][name] * r for name, r in spectra.items())
        spectra_xml = "\n".join(f'''				<spectrum type="piecewise_linear" id="{name}">
{spectrum_xml(nm, r, 5)}
				</spectrum>''' for name, r in spectra.items())
        print(f"materials: {', '.join(spectra)} ({materials_path.name}); mean reflectance at 550 nm "
              f"{np.interp(550, nm, mean):.3f}")
        return f'''<!-- materials: {", ".join(f"{m['name']} ({m['source']})" for m in materials["materials"])} -->
		<bsdf type="diffuse">
			<texture name="reflectance" type="spectral_mix">
				<texture name="weights" type="bitmap">
					<path value="{materials["weights"]}"/>
					<string name="filter_type" value="bilinear"/>
					<string name="wrap_mode" value="clamp"/>
				</texture>
{spectra_xml}
			</texture>
		</bsdf>''', mean
    median = placeholder_reflectance if placeholder_reflectance is not None else 0.35
    mean = reflectance_map(naip_path, site_dir / "reflectance.png", median)
    print(f"placeholder reflectance: median {median:g}, mean {mean:.3f}")
    return '''<bsdf type="diffuse">
			<texture name="reflectance" type="bitmap">
				<path value="reflectance.png"/>
				<string name="colour_encoding" value="linear"/>
				<string name="filter_type" value="bilinear"/>
				<string name="wrap_mode" value="clamp"/>
			</texture>
		</bsdf>''', np.full(len(nm), mean)


def class_bsdfs(materials, nm):
    """Of materials.json of classes (materials.py --classes): each class's diffuse BSDF (XML; the wall's, of its one
    material, is uniform), and the site's mean reflectance, at wavelengths `nm`."""
    wl = np.asarray(materials["wavelength_nm"])
    if nm[0] < wl[0] or nm[-1] > wl[-1]:
        raise SystemExit(f"materials.json covers {wl[0]:g}-{wl[-1]:g} nm, not {nm[0]:g}-{nm[-1]:g} nm")
    bsdfs = {}
    for c in materials["classes"] + ([materials["wall"]] if materials.get("wall") else []):
        spectra = {name: np.interp(nm, wl, r) for name, r in c["reflectance"].items()}
        sources = ", ".join(f"{m['name']} ({m['source']})" for m in c["materials"])
        if c["weights"] is None:
            (name, r), = spectra.items()
            bsdfs[c["name"]] = f'''<!-- {c["name"]}: {sources} -->
\t\t<bsdf type="diffuse">
\t\t\t<spectrum name="reflectance" type="piecewise_linear">
{spectrum_xml(nm, r, 4)}
\t\t\t</spectrum>
\t\t</bsdf>'''
            continue
        spectra_xml = "\n".join(f'''\t\t\t\t<spectrum type="piecewise_linear" id="{c["name"]}_{name}">
{spectrum_xml(nm, r, 5)}
\t\t\t\t</spectrum>''' for name, r in spectra.items())
        bsdfs[c["name"]] = f'''<!-- {c["name"]}: {sources} -->
\t\t<bsdf type="diffuse">
\t\t\t<texture name="reflectance" type="spectral_mix">
\t\t\t\t<texture name="weights" type="bitmap">
\t\t\t\t\t<path value="{c["weights"]}"/>
\t\t\t\t\t<string name="filter_type" value="bilinear"/>
\t\t\t\t\t<string name="wrap_mode" value="clamp"/>
\t\t\t\t</texture>
{spectra_xml}
\t\t\t</texture>
\t\t</bsdf>'''
    mean = np.interp(nm, wl, materials["mean_reflectance"])
    print(f"materials: classes {', '.join(bsdfs)}; mean reflectance at 550 nm {np.interp(550, nm, mean):.3f}")
    return bsdfs, mean


def solar_spectrum(csv_path, nm):
    """TSIS-1 HSRS at 1 AU, W/m²/µm, at wavelengths `nm` (1 nm bins: linear interpolation)."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=(0, 1))
    return np.interp(nm, data[:, 0], data[:, 1]) * 1e3


def spectrum_xml(nm, values, tabs):
    """Bins of a piecewise-linear spectrum, indented by `tabs`."""
    return "\n".join("\t" * tabs + f'<bin wavelength="{w:g}nm" value="{v:.6g}"/>' for w, v in zip(nm, values))


def load_atmosphere(path, nm, zenith, view_zenith, view_azimuth):
    """atmosphere.json's quantities at wavelengths `nm`; checks that it was computed for this geometry."""
    atm = json.loads(path.read_text())
    if (abs(atm["sun"]["zenith_deg"] - zenith) > 0.01 or atm["view"]["zenith_deg"] != view_zenith
            or (view_zenith != 0 and atm["view"]["azimuth_deg"] != view_azimuth)):
        raise SystemExit(f"{path} is for the sun at zenith {atm['sun']['zenith_deg']:.3f}° and view zenith "
                         f"{atm['view']['zenith_deg']}°, azimuth {atm['view']['azimuth_deg']}°: run atmosphere.py "
                         "for this time and view, or give --no-atmosphere")
    wl = np.asarray(atm["wavelength_nm"])
    if nm[0] < wl[0] or nm[-1] > wl[-1]:
        raise SystemExit(f"{path} covers {wl[0]:g}-{wl[-1]:g} nm, not {nm[0]:g}-{nm[-1]:g} nm")
    keys = [k for k, v in atm.items() if isinstance(v, list) and len(v) == len(wl) and k != "wavelength_nm"]
    return atm, {k: np.interp(nm, wl, atm[k]) for k in keys}


def sky_dome(path, radius, to_sun, hole_deg, step_deg=1.0):
    """A hemisphere of `radius` m about the origin, above the horizon, facing inwards, without the triangles within
    `hole_deg` of the direction `to_sun`. Returns its projected solid angle seen from a horizontal surface at the
    origin, sr (π, less the hole's)."""
    rings = int(round(90 / step_deg))
    sectors = int(round(360 / step_deg))
    el = np.radians(np.arange(rings) * step_deg)
    az = np.radians(np.arange(sectors) * step_deg)
    E, A = np.meshgrid(el, az, indexing="ij")
    dirs = np.stack([np.cos(E) * np.sin(A), np.sin(E), -np.cos(E) * np.cos(A)], -1).reshape(-1, 3)
    dirs = np.concatenate([dirs, [[0, 1, 0]]])               # the zenith
    top = len(dirs) - 1
    idx = np.arange(rings * sectors).reshape(rings, sectors)
    a, b = idx[:-1], np.roll(idx[:-1], -1, axis=1)          # ring r, sectors s and s+1
    c, d = idx[1:], np.roll(idx[1:], -1, axis=1)            # ring r+1
    # wound so that their normals, (v1-v0)×(v2-v0), face the origin
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([b, d, c], -1).reshape(-1, 3),
                            np.stack([idx[-1], np.roll(idx[-1], -1), np.full(sectors, top)], -1)])
    v = dirs[faces]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    assert (np.einsum("ij,ij->i", n, v.mean(axis=1)) < 0).all()
    s = np.asarray(to_sun) / np.linalg.norm(to_sun)
    keep = ~(v @ s > math.cos(math.radians(hole_deg + step_deg))).any(axis=1)   # clear within hole_deg
    faces = faces[keep]
    # Lambert's formula: a spherical polygon's projected solid angle is ½ Σ θ_i n·(v_i×v_i+1)/|v_i×v_i+1|
    v = dirs[faces]
    proj = np.zeros(len(faces))
    for i in range(3):
        p, q = v[:, i], v[:, (i + 1) % 3]
        cr = np.cross(p, q)
        proj += 0.5 * np.arccos(np.clip(np.einsum("ij,ij->i", p, q), -1, 1)) * cr[:, 1] / np.linalg.norm(cr, axis=1)
    normals = -dirs
    uvs = np.zeros((len(dirs), 2))
    write_ply(path, dirs * radius, normals, uvs, faces)
    return np.abs(proj).sum(), len(faces)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("site_dir", type=pathlib.Path)
    p.add_argument("--name", default="scene", help="scene file name (default scene.xml)")
    p.add_argument("--time", type=datetime.datetime.fromisoformat, default=datetime.datetime(2022, 7, 19, 18, 19, 19, tzinfo=datetime.timezone.utc),
                   help="UTC time, for the sun (default: Sentinel-2B's pass, 2022-07-19T18:19:19Z)")
    p.add_argument("--altitude", type=float, default=500e3, help="camera altitude, m (default 500 km)")
    p.add_argument("--view-zenith", type=float, default=0, help="degrees (default 0: nadir)")
    p.add_argument("--view-azimuth", type=float, default=0, help="degrees clockwise from north, towards the camera")
    p.add_argument("--gsd", type=float, default=1.0, help="ground sampling distance at nadir, m (default 1)")
    p.add_argument("--pixel-pitch", type=float, default=5e-6, help="m (default 5 µm)")
    p.add_argument("--pixels", type=int, default=1920, help="film width and height (default 1920)")
    p.add_argument("--spp", type=int, default=16, help="samples per pixel (default 16)")
    p.add_argument("--ray-tracing", action="store_true", help="trace rays rather than beams")
    p.add_argument("--surface", choices=["dsm", "dem"],
                   help="the terrain's heights: the lidar surface model (buildings, trees; the default if the site "
                        "has one) or the bare-earth DEM")
    p.add_argument("--mesh-spacing", type=int,
                   help="height map pixels between mesh vertices (default 1 for the surface model, 2 for the DEM)")
    p.add_argument("--reflectance", type=float,
                   help="use a placeholder reflectance map of this median rather than SITE_DIR/materials.json "
                        "(default, without it: .35)")
    p.add_argument("--fsd", action=argparse.BooleanOptionalAction,
                   help="free-space diffraction by the terrain's edges (beams only; default: on for the DEM, off for "
                        "the surface model, whose many sharp edges make it slow and spiky, while diffraction by "
                        "metre-sized edges is negligible at these wavelengths)")
    p.add_argument("--no-atmosphere", action="store_true", help="ignore SITE_DIR/atmosphere.json")
    p.add_argument("--sky-radius", type=float, default=100e3, help="m (default 100 km)")
    p.add_argument("--telescope", choices=["cassegrain", "ritchey-chretien", "three-mirror"],
                   help="image through a telescope's mirrors, traced (the telescope sensor, of telescope.py's two-mirror "
                        "design, or its three-mirror anastigmat, at the camera's focal length) rather than a pinhole "
                        "camera (satellite_frame)")
    p.add_argument("--telescope-off-axis", type=float,
                   help="m: an off-axis segment of the design, its aperture centred this far from the axis (two-mirror: "
                        "default on axis; three-mirror: default the aperture's diameter)")
    p.add_argument("--field-bias", type=float,
                   help="degrees: the film's centre this far off the telescope's axis, towards north (two-mirror: "
                        "default 0; three-mirror: default 1)")
    p.add_argument("--curved-focal-plane", action="store_true",
                   help="the telescope's focal plane curved: the sphere on which the film's field images best "
                        "(telescope.py)")
    p.add_argument("--baffles", action="store_true",
                   help="an on-axis two-mirror telescope's baffles: a tube before the secondary, a vane on the primary "
                        "(telescope.py)")
    p.add_argument("--barrel", action="store_true",
                   help="enclose the telescope in a tube (telescope.py): else its focal plane sees the scene around its "
                        "primary's rim")
    p.add_argument("--stray-light", action="store_true",
                   help="trace the telescope non-sequentially (its stray light)")
    p.add_argument("--surroundings", type=float,
                   help="m: ground beyond the terrain, out to this distance from its centre, flat, of the terrain's mean "
                        "reflectance (the stray light from outside the field sees it); the sky is made large enough to "
                        "cover it, its hole for the sun to light it")
    p.add_argument("--telescope-aperture", type=float, default=0.4, help="the telescope's diameter, m (default 0.4)")
    p.add_argument("--primary-f-number", type=float, default=1.5,
                   help="the telescope's primary's focal length over the aperture (default 1.5)")
    p.add_argument("--back-focus", type=float,
                   help="the telescope's focal plane behind its primary, m (default 0.375 times the aperture)")
    p.add_argument("--coating", choices=list(telescope.COATINGS), default="ideal",
                   help="the telescope's mirrors' (default ideal)")
    args = p.parse_args()

    site = json.loads((args.site_dir / "site.json").read_text())
    west, south, east, north = site["bounds"]
    centre_en = ((west + east) / 2, (south + north) / 2)
    centre_h = site["centre"]["elevation_m"]

    surface = args.surface or ("dsm" if site.get("lidar") else "dem")
    if surface == "dsm" and not site.get("lidar"):
        raise SystemExit("--surface dsm needs the site's lidar surface model: fetch_site.py --lidar")
    height_map = site["lidar"]["dsm"] if surface == "dsm" else site["dem"]["file"]
    spacing = args.mesh_spacing or (1 if surface == "dsm" else 2)
    vertices, normals, E, N, faces = terrain(args.site_dir / height_map, spacing, centre_en, centre_h)
    naip_path = args.site_dir / site["naip"]["file"]
    with rasterio.open(naip_path) as ds:
        tex_bounds = ds.bounds      # the materials' weights and reflectance.png are on NAIP's grid
    uvs = np.stack([(E - tex_bounds.left) / (tex_bounds.right - tex_bounds.left),
                    (N - tex_bounds.bottom) / (tex_bounds.top - tex_bounds.bottom)], axis=-1)
    print(f"terrain: {height_map}, {len(vertices)} vertices, {len(faces)} triangles; relief "
          f"{vertices[:, 1].min():.2f} .. {vertices[:, 1].max():.2f} m")
    # the surface model's edges (walls, crowns) are sharp: flat triangles; the DEM is smooth: vertex normals
    face_normals = surface == "dsm"
    fsd = args.fsd if args.fsd is not None else surface != "dsm"
    materials_path = args.site_dir / "materials.json"
    materials = json.loads(materials_path.read_text()) if materials_path.exists() and args.reflectance is None else None
    classes = materials is not None and "classes" in materials
    if classes:
        names = [c["name"] for c in materials["classes"]] + (["wall"] if materials.get("wall") else [])
        fc = face_classes(vertices, E, N, faces, materials, args.site_dir, tex_bounds)
        meshes = write_class_meshes(args.site_dir, vertices, uvs, faces, fc, names)
        print("terrain classes: " + ", ".join(f"{n} {k / len(faces):.1%}" for n, k in meshes))
    else:
        write_ply(args.site_dir / "terrain.ply", vertices, None if face_normals else normals, uvs, faces)

    zen, az = solar_position(args.time, site["centre"]["lat"], site["centre"]["lon"])
    d = earth_sun_distance(args.time.date())
    print(f"sun at {args.time.isoformat()}: zenith {zen:.3f}°, azimuth {az:.3f}°; Earth-Sun distance {d:.5f} AU")
    z, a = math.radians(zen), math.radians(az)
    to_sun = (math.sin(z) * math.sin(a), math.cos(z), -math.sin(z) * math.cos(a))
    focal_length = args.altitude * args.pixel_pitch / args.gsd
    telescope_design = None
    if not args.telescope and (args.telescope_off_axis is not None or args.field_bias is not None
                               or args.curved_focal_plane or args.baffles or args.barrel or args.stray_light):
        raise SystemExit("--telescope-off-axis, --field-bias, --curved-focal-plane, --baffles, --barrel and "
                         "--stray-light are options of --telescope")
    if args.telescope:
        D = args.telescope_aperture
        field = math.atan(args.pixels * args.pixel_pitch / math.sqrt(2) / focal_length)
        if args.telescope == "three-mirror":
            # telescope.py's layout (for f = 2.5 m: f1 0.6 m, separations 0.45 and 1.15 m), scaled to the focal length
            f = focal_length
            design = telescope.three_mirror(
                D, f, 0.24 * f, (0.18 * f, 0.46 * f),
                off_axis=args.telescope_off_axis if args.telescope_off_axis is not None else D,
                field_bias=math.radians(args.field_bias if args.field_bias is not None else 1.0), field_radius=field)
        else:
            design = telescope.two_mirror(
                args.telescope, D, focal_length, args.primary_f_number * D,
                args.back_focus if args.back_focus is not None else 0.375 * D, field_radius=field,
                off_axis=args.telescope_off_axis or 0.0, field_bias=math.radians(args.field_bias or 0.0))
        if args.curved_focal_plane:
            design = telescope.curve_focal_plane(design, field)
        if args.baffles:
            design = telescope.baffle(design)
        if args.barrel:
            design = telescope.barrel(design, field_radius=field + math.radians(abs(args.field_bias or 0.0)))
        telescope_design = telescope.coat(design, args.coating)
        print(f"telescope: {telescope_design.name}, D {D:g} m, obscuration {telescope_design.obscuration:.3f}, "
              f"{args.coating} mirrors" + (f", focal plane radius {design.focal_plane_radius:.4g} m"
                                            if design.focal_plane_radius else "")
              + (", baffles" if args.baffles else "") + (", barrel" if args.barrel else "")
              + (", stray light" if args.stray_light else ""))
    # the telescope sensor's optics (telescope.py); the focal length it checks them against: the image's scale about
    # the film's centre (a biased field's differs from the design's focal length)
    telescope_xml = "" if telescope_design is None else "\n" + telescope.elements_xml(telescope_design, indent="\t\t")
    if args.stray_light:
        telescope_xml = '\n\t\t<boolean name="stray_light" value="true"/>' + telescope_xml
    sensor_focal_length = focal_length if telescope_design is None else \
        telescope_design.film_centre_scale or telescope_design.focal_length

    band_xml = "\n".join(
        f'''				<spectrum type="piecewise_linear" id="{name}">
					<bin wavelength="{lo}nm" value="1"/>
					<bin wavelength="{hi}nm" value="1"/>
				</spectrum>''' for name, (lo, hi) in BANDS.items())
    lo_nm = min(lo for lo, _ in BANDS.values()) - 10
    hi_nm = max(hi for _, hi in BANDS.values()) + 10
    nm = np.arange(lo_nm, hi_nm + 1.0)
    E0 = solar_spectrum(args.site_dir / site["solar_spectrum"]["file"], nm) / d**2
    mus = math.cos(z)
    in_band = {name: (nm >= lo) & (nm <= hi) for name, (lo, hi) in BANDS.items()}
    if classes:
        bsdfs, mean_reflectance = class_bsdfs(materials, nm)
        shapes = [(f"terrain_{name}.ply", bsdfs[name]) for name, _ in meshes]
    else:
        bsdf_xml, mean_reflectance = terrain_bsdf(args.site_dir, nm, args.reflectance, naip_path)
        shapes = [("terrain.ply", bsdf_xml)]
    terrain_xml = "".join(f'''
\t<shape type="ply">
\t\t<path value="{path}"/>
\t\t<quantity name="scale" value="1m"/>
\t\t<boolean name="face_normals" value="{str(face_normals).lower()}"/>
\t\t{bsdf_xml}
\t</shape>
''' for path, bsdf_xml in shapes)

    atm_path = args.site_dir / "atmosphere.json"
    atm = None
    if not args.no_atmosphere and atm_path.exists():
        atm_info, atm = load_atmosphere(atm_path, nm, zen, args.view_zenith, args.view_azimuth)
    sidecar = {"bands_nm": BANDS, "gsd_m": args.gsd, "pixels": args.pixels, "view_zenith_deg": args.view_zenith,
               "view_azimuth_deg": args.view_azimuth, "altitude_m": args.altitude, "focal_length_m": focal_length,
               "pixel_pitch_m": args.pixel_pitch, "time": args.time.isoformat(), "atmosphere": None}
    if telescope_design is not None:
        # what detector.py needs of the render's telescope: its aperture, and the film centre's projected solid angle,
        # π (sin²U - sin²u) of the pupil's outer and inner radii h, sin U = h/f (Abbe's sine condition)
        D, eps = telescope_design.entrance_pupil_diameter, telescope_design.obscuration
        sidecar["telescope"] = {
            "design": telescope_design.name, "coating": args.coating, "aperture_diameter_m": D, "obscuration": eps,
            "projected_solid_angle_sr": math.pi * (D / 2 / focal_length) ** 2 * (1 - eps**2),
            "stray_light": args.stray_light,
            # the mirrors' reflectance per band: toa.py's path radiance, added to the render, must pass them too
            "band_throughput": {band: telescope.band_throughput(telescope_design, nm) for band, nm in BANDS.items()}}
    if atm is None:
        print("no atmosphere" + ("" if args.no_atmosphere else f" ({atm_path} not found)"))
        sun_irradiance = E0
        sky_xml = ""
        transmittance_xml = ""
        radiance_where = "at the top of a (missing) atmosphere"
    else:
        # the sun, through the atmosphere
        sun_irradiance = E0 * atm["T_down_direct"] * atm["Tg_down"]
        # the sky: diffuse light, and light reflected between the ground and the atmosphere
        rho_e, S = mean_reflectance, atm["spherical_albedo"]
        sky_irradiance = E0 * mus * atm["Tg_down"] * (atm["T_down_total"] / (1 - rho_e * S) - atm["T_down_direct"])
        # a hole wide enough for the sun to reach every point of the scene (its surroundings too: the sky 20 times as
        # far, so that the hole stays small)
        half_diagonal = np.hypot(np.abs(vertices[:, 0]).max(), np.abs(vertices[:, 2]).max())
        sky_radius = args.sky_radius
        if args.surroundings:
            half_diagonal = max(half_diagonal, args.surroundings * math.sqrt(2))
            sky_radius = max(sky_radius, 20 * half_diagonal)
        hole_deg = math.degrees(math.atan2(half_diagonal, sky_radius)) + 0.5
        # (a larger sky is the scene's own: other scenes of the site share sky.ply)
        sky_file = f"sky_{args.name}.ply" if sky_radius != args.sky_radius else "sky.ply"
        sky_projected_solid_angle, sky_triangles = sky_dome(args.site_dir / sky_file, sky_radius, to_sun, hole_deg)
        sky_radiance = sky_irradiance / sky_projected_solid_angle
        print(f"atmosphere: {atm_info['source']}; sky: {sky_triangles} triangles, radius {sky_radius / 1e3:g} km, "
              f"hole {hole_deg:.2f}°, "
              f"projected solid angle π - {math.pi - sky_projected_solid_angle:.2e} sr")
        sky_xml = f'''
	<!-- the sky, of uniform radiance (W/m²/sr/µm) -->
	<shape type="ply">
		<path value="{sky_file}"/>
		<quantity name="scale" value="1m"/>
		<bsdf type="diffuse">
			<spectrum name="reflectance" constant="0"/>
		</bsdf>
		<emitter type="area">
			<spectrum name="radiance" type="piecewise_linear">
{spectrum_xml(nm, sky_radiance, 4)}
			</spectrum>
		</emitter>
	</shape>
'''
        # the camera sees the ground through the direct transmittance, and the rest of the gases' two-way transmittance
        transmittance = atm["T_up_direct"] * atm["Tg_total"] / atm["Tg_down"]
        transmittance_xml = f'''
				<spectrum name="transmittance" type="piecewise_linear">
{spectrum_xml(nm, transmittance, 5)}
				</spectrum>'''
        radiance_where = "at the top of the atmosphere, straight from the ground (toa.py adds the rest)"

        # for toa.py, band-averaged: the path radiance; the ratio of diffuse to direct upward transmittance, weighted by
        # the spectrum of the ground's radiance (as for a grey ground), and how much of it is from molecules
        weight = E0 * atm["Tg_total"] * atm["T_down_total"] * atm["T_up_direct"] / (1 - rho_e * S)
        diffuse = atm["T_up_total"] - atm["T_up_direct"]
        r_mol, r_aer = atm["T_up_diffuse_rayleigh"], atm["T_up_diffuse_aerosol"]
        sidecar["atmosphere"] = {
            "source": atm_info["source"], "inputs": atm_info["inputs"],
            "mean_reflectance": {b: float(np.mean(rho_e[m])) for b, m in in_band.items()},
            "path_radiance": {b: float(np.mean((atm["rho_atm"] * E0 * mus / math.pi)[m])) for b, m in in_band.items()},
            "diffuse_to_direct_up": {b: float(np.sum((weight * diffuse / atm["T_up_direct"])[m]) / np.sum(weight[m]))
                                     for b, m in in_band.items()},
            "rayleigh_fraction": {b: float(np.sum((weight * r_mol)[m]) / np.sum((weight * (r_mol + r_aer))[m]))
                                  for b, m in in_band.items()},
        }

    surroundings_xml = ""
    if args.surroundings:
        # a square beneath the terrain's lowest point (which hides it within the terrain), of its mean reflectance
        R, y = args.surroundings, float(vertices[:, 1].min()) - 0.5
        surroundings_xml = f'''
	<!-- the surroundings: flat ground, {R / 1e3:g} km about the centre, of the terrain's mean reflectance -->
	<shape type="rectangle">
		<point name="p" value="{-R:g}m, {y:.3f}m, {R:g}m"/>
		<point name="x" value="{2 * R:g}m, 0m, 0m"/>
		<point name="y" value="0m, 0m, {-2 * R:g}m"/>
		<bsdf type="diffuse">
			<spectrum name="reflectance" type="piecewise_linear">
{spectrum_xml(nm, mean_reflectance, 4)}
			</spectrum>
		</bsdf>
	</shape>
'''
        print(f"surroundings: {2 * R / 1e3:g} km square at {y:.2f} m")

    lidar_note = f"; surface: 3DEP lidar, {site['lidar']['collection']}" if surface == "dsm" else ""
    xml = f'''<?xml version="1.0" encoding="utf-8"?>
<!-- Generated by tools/remote_sensing/build_scene.py from {args.site_dir.name}: {site["size_m"]:g} m around
     {site["centre"]["lat"]}, {site["centre"]["lon"]} ({site["dem"]["source"]}, {site["dem"]["project"]}{lidar_note}; NAIP {", ".join(site["naip"]["dates"])}).
     Frame: x east, y up, z south, metres, origin on the ground at the centre ({centre_h:.2f} m NAVD88).
     Camera: {args.altitude / 1e3:g} km, view zenith {args.view_zenith:g}°, azimuth {args.view_azimuth:g}°, {args.gsd:g} m GSD at nadir{'' if telescope_design is None else f', through a {telescope_design.name} telescope ({args.coating} mirrors)'}.
     Pixels record band-averaged radiance, W/m²/sr/µm, {radiance_where}.
     Needs wave_tracer built with REALISTIC_REMOTE_SENSING. -->
<scene version="0.1.0">
	<default name="spp" value="{args.spp}"/>
	<default name="ray_tracing" value="{str(args.ray_tracing).lower()}"/>

	<integrator type="plt_path">
		<integer name="max_depth" value="4"/>
		<string name="direction" value="backward"/>
		<boolean name="FSD" value="{str(fsd).lower()}"/>
	</integrator>

	        <sensor type="{'satellite_frame' if telescope_design is None else 'telescope'}" id="camera">
		                <quantity name="altitude" value="{args.altitude:g}m"/>
		                <quantity name="view_zenith" value="{args.view_zenith:g}°"/>
		                <quantity name="view_azimuth" value="{args.view_azimuth:g}°"/>
		                <quantity name="focal_length" value="{sensor_focal_length:.12g}m"/>
		                <quantity name="pixel_pitch" value="{args.pixel_pitch:.12g}m"/>
		                <boolean name="ray_trace_only" value="$ray_tracing"/>{telescope_xml}
		<integer name="samples" value="$spp"/>
		<film type="array">
			<integer name="width" value="{args.pixels}"/>
			<integer name="height" value="{args.pixels}"/>
			<response type="band_radiance">
				<string name="spectral_unit" value="um"/>
				<boolean name="normalize" value="true"/>
{band_xml}{transmittance_xml}
			</response>
		</film>
	</sensor>

	<!-- the sun: TSIS-1 HSRS (W/m²/µm), at {d:.5f} AU{'' if atm is None else ', through the atmosphere'} -->
	<emitter type="directional">
		<transform name="to_world">
			<lookat origin="0m, 0m, 0m" target="{-to_sun[0]:.9f}m, {-to_sun[1]:.9f}m, {-to_sun[2]:.9f}m" up="0, 1, 0"/>
		</transform>
		<spectrum name="irradiance" type="piecewise_linear">
{spectrum_xml(nm, sun_irradiance, 3)}
		</spectrum>
	</emitter>
{sky_xml}{terrain_xml}{surroundings_xml}</scene>
'''
    (args.site_dir / f"{args.name}.xml").write_text(xml)
    (args.site_dir / f"{args.name}.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    print(f"wrote {args.site_dir}/{', '.join(path for path, _ in shapes)}, {sky_file + ', ' if atm is not None else ''}"
          f"{args.name}.xml, "
          f"{args.name}.json; camera focal length {focal_length:g} m")


if __name__ == "__main__":
    main()
