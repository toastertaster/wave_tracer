# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "rasterio", "pillow"]
# ///
"""Builds a wave_tracer scene of a site fetched by fetch_site.py.

Writes into the site directory:
  terrain.ply      the DEM as a triangle mesh, with vertex normals and texture
                   coordinates
  reflectance.png  without materials.json: a 16-bit linear reflectance map
  sky.ply          with an atmosphere: the sky, a hemisphere around the scene
  <name>.xml       the scene: the terrain, the sun, the sky, and a
                   satellite_frame camera recording band radiance (needs a
                   wave_tracer built with REALISTIC_REMOTE_SENSING)
  <name>.json      what toa.py and detector.py need: the bands, the camera,
                   and the atmosphere's own radiance

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
abundances (materials.exr). Without them, or with --reflectance, a
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
import struct

import numpy as np
import rasterio
from PIL import Image

from sun import earth_sun_distance, solar_position

# placeholder multispectral bands (nm), as boxes: those of typical very-high-resolution imagers
BANDS = {"blue": (450, 520), "green": (520, 600), "red": (630, 690), "nir": (760, 900)}


def write_ply(path, vertices, normals, uvs, faces):
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(vertices)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property float nx\nproperty float ny\nproperty float nz\n"
                 "property float u\nproperty float v\n"
                 f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        v = np.concatenate([vertices, normals, uvs], axis=1).astype("<f4")
        f.write(v.tobytes())
        fc = np.empty(len(faces), dtype=[("n", "u1"), ("i", "<i4", (3,))])
        fc["n"] = 3
        fc["i"] = faces
        f.write(fc.tobytes())


def terrain(dem_path, spacing: int, centre_en, centre_h):
    """The DEM, sampled every `spacing` pixels (at pixel centres), as a mesh in the scene frame."""
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
    p.add_argument("--mesh-spacing", type=int, default=2, help="DEM pixels between mesh vertices (default 2)")
    p.add_argument("--reflectance", type=float,
                   help="use a placeholder reflectance map of this median rather than SITE_DIR/materials.json "
                        "(default, without it: .35)")
    p.add_argument("--no-atmosphere", action="store_true", help="ignore SITE_DIR/atmosphere.json")
    p.add_argument("--sky-radius", type=float, default=100e3, help="m (default 100 km)")
    args = p.parse_args()

    site = json.loads((args.site_dir / "site.json").read_text())
    west, south, east, north = site["bounds"]
    centre_en = ((west + east) / 2, (south + north) / 2)
    centre_h = site["centre"]["elevation_m"]

    vertices, normals, E, N, faces = terrain(args.site_dir / site["dem"]["file"], args.mesh_spacing, centre_en, centre_h)
    naip_path = args.site_dir / site["naip"]["file"]
    with rasterio.open(naip_path) as ds:
        tex_bounds = ds.bounds      # materials.exr and reflectance.png are on NAIP's grid
    uvs = np.stack([(E - tex_bounds.left) / (tex_bounds.right - tex_bounds.left),
                    (N - tex_bounds.bottom) / (tex_bounds.top - tex_bounds.bottom)], axis=-1)
    write_ply(args.site_dir / "terrain.ply", vertices, normals, uvs, faces)
    print(f"terrain: {len(vertices)} vertices, {len(faces)} triangles; relief {vertices[:, 1].min():.2f} .. {vertices[:, 1].max():.2f} m")

    zen, az = solar_position(args.time, site["centre"]["lat"], site["centre"]["lon"])
    d = earth_sun_distance(args.time.date())
    print(f"sun at {args.time.isoformat()}: zenith {zen:.3f}°, azimuth {az:.3f}°; Earth-Sun distance {d:.5f} AU")
    z, a = math.radians(zen), math.radians(az)
    to_sun = (math.sin(z) * math.sin(a), math.cos(z), -math.sin(z) * math.cos(a))
    focal_length = args.altitude * args.pixel_pitch / args.gsd

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
    terrain_xml, mean_reflectance = terrain_bsdf(args.site_dir, nm, args.reflectance, naip_path)

    atm_path = args.site_dir / "atmosphere.json"
    atm = None
    if not args.no_atmosphere and atm_path.exists():
        atm_info, atm = load_atmosphere(atm_path, nm, zen, args.view_zenith, args.view_azimuth)
    sidecar = {"bands_nm": BANDS, "gsd_m": args.gsd, "pixels": args.pixels, "view_zenith_deg": args.view_zenith,
               "view_azimuth_deg": args.view_azimuth, "altitude_m": args.altitude, "focal_length_m": focal_length,
               "pixel_pitch_m": args.pixel_pitch, "time": args.time.isoformat(), "atmosphere": None}
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
        # a hole wide enough for the sun to reach every point of the scene
        half_diagonal = np.hypot(np.abs(vertices[:, 0]).max(), np.abs(vertices[:, 2]).max())
        hole_deg = math.degrees(math.atan2(half_diagonal, args.sky_radius)) + 0.5
        sky_projected_solid_angle, sky_triangles = sky_dome(args.site_dir / "sky.ply", args.sky_radius, to_sun, hole_deg)
        sky_radiance = sky_irradiance / sky_projected_solid_angle
        print(f"atmosphere: {atm_info['source']}; sky: {sky_triangles} triangles, hole {hole_deg:.2f}°, "
              f"projected solid angle π - {math.pi - sky_projected_solid_angle:.2e} sr")
        sky_xml = f'''
	<!-- the sky, of uniform radiance (W/m²/sr/µm) -->
	<shape type="ply">
		<path value="sky.ply"/>
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

    xml = f'''<?xml version="1.0" encoding="utf-8"?>
<!-- Generated by tools/remote_sensing/build_scene.py from {args.site_dir.name}: {site["size_m"]:g} m around
     {site["centre"]["lat"]}, {site["centre"]["lon"]} ({site["dem"]["source"]}, {site["dem"]["project"]}; NAIP {", ".join(site["naip"]["dates"])}).
     Frame: x east, y up, z south, metres, origin on the ground at the centre ({centre_h:.2f} m NAVD88).
     Camera: {args.altitude / 1e3:g} km, view zenith {args.view_zenith:g}°, azimuth {args.view_azimuth:g}°, {args.gsd:g} m GSD at nadir.
     Pixels record band-averaged radiance, W/m²/sr/µm, {radiance_where}.
     Needs wave_tracer built with REALISTIC_REMOTE_SENSING. -->
<scene version="0.1.0">
	<default name="spp" value="{args.spp}"/>
	<default name="ray_tracing" value="{str(args.ray_tracing).lower()}"/>

	<integrator type="plt_path">
		<integer name="max_depth" value="4"/>
		<string name="direction" value="backward"/>
	</integrator>

	<sensor type="satellite_frame" id="camera">
		<quantity name="altitude" value="{args.altitude:g}m"/>
		<quantity name="view_zenith" value="{args.view_zenith:g}°"/>
		<quantity name="view_azimuth" value="{args.view_azimuth:g}°"/>
		<quantity name="focal_length" value="{focal_length:g}m"/>
		<quantity name="pixel_pitch" value="{args.pixel_pitch:g}m"/>
		<boolean name="ray_trace_only" value="$ray_tracing"/>
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
{sky_xml}
	<shape type="ply">
		<path value="terrain.ply"/>
		<quantity name="scale" value="1m"/>
		{terrain_xml}
	</shape>
</scene>
'''
    (args.site_dir / f"{args.name}.xml").write_text(xml)
    (args.site_dir / f"{args.name}.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    print(f"wrote {args.site_dir}/terrain.ply, {'sky.ply, ' if atm is not None else ''}{args.name}.xml, "
          f"{args.name}.json; camera focal length {focal_length:g} m")


if __name__ == "__main__":
    main()
