# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "rasterio", "pillow"]
# ///
"""Builds a wave_tracer scene of a site fetched by fetch_site.py.

Writes into the site directory:
  terrain.ply      the DEM as a triangle mesh, with vertex normals and texture
                   coordinates
  reflectance.png  a 16-bit linear reflectance map
  <name>.xml       the scene: the terrain, the sun, and a satellite_frame camera
                   recording band radiance (needs a wave_tracer built with
                   REALISTIC_REMOTE_SENSING)

The scene's frame is local to the site: x east, y up, z south (north is -z),
in metres, with the origin on the ground at the site's centre. The camera is
placed by altitude and view angles over that origin (see the satellite_frame
sensor). The sun shines from its position at the given time, with the TSIS-1
top-of-atmosphere spectrum scaled to the Earth-Sun distance of that day.
There is no atmosphere yet.

Reflectance is a placeholder until spectral material maps: spectrally flat,
scaled from NAIP's brightness so that its median is --reflectance. (NAIP is not
radiometrically calibrated; its RGB would also leave the near infrared dark.)

    uv run tools/remote_sensing/build_scene.py SITE_DIR [--gsd 1 --pixels 1920 --altitude 500e3 ...]

Render it with
    wave_tracer render [--gpu] SITE_DIR/scene.xml -o OUT_DIR
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

# placeholder multispectral bands (nm), as boxes: those of typical very-high-resolution imagers
BANDS = {"blue": (450, 520), "green": (520, 600), "red": (630, 690), "nir": (760, 900)}


def solar_position(when: datetime.datetime, lat: float, lon: float):
    """Solar zenith and azimuth (clockwise from north), degrees: NOAA's solar position algorithm."""
    jd = when.timestamp() / 86400 + 2440587.5
    t = (jd - 2451545) / 36525
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    mr = math.radians(m)
    c = (math.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t)) + math.sin(2 * mr) * (0.019993 - 0.000101 * t)
         + math.sin(3 * mr) * 0.000289)
    omega = 125.04 - 1934.136 * t
    app_lon = l0 + c - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    obliq = 23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60) / 60
    obliq += 0.00256 * math.cos(math.radians(omega))
    decl = math.asin(math.sin(math.radians(obliq)) * math.sin(math.radians(app_lon)))
    y = math.tan(math.radians(obliq) / 2) ** 2
    l0r = math.radians(l0)
    eqtime = 4 * math.degrees(y * math.sin(2 * l0r) - 2 * e * math.sin(mr) + 4 * e * y * math.sin(mr) * math.cos(2 * l0r)
                              - 0.5 * y * y * math.sin(4 * l0r) - 1.25 * e * e * math.sin(2 * mr))
    minutes = when.hour * 60 + when.minute + when.second / 60
    hour_angle = math.radians((minutes + eqtime + 4 * lon) / 4 - 180)
    latr = math.radians(lat)
    cos_zen = math.sin(latr) * math.sin(decl) + math.cos(latr) * math.cos(decl) * math.cos(hour_angle)
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    az = math.atan2(math.sin(hour_angle), math.cos(hour_angle) * math.sin(latr) - math.tan(decl) * math.cos(latr))
    return math.degrees(zen), (math.degrees(az) + 180) % 360


def earth_sun_distance(day: datetime.date) -> float:
    """Earth-Sun distance, AU (to ~1e-4)."""
    g = math.radians(357.529 + 0.98560028 * (day.toordinal() - datetime.date(2000, 1, 1).toordinal() - 0.5 + 1))
    return 1.00014 - 0.01671 * math.cos(g) - 0.00014 * math.cos(2 * g)


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
    """NAIP's brightness (mean of red, green, blue), scaled so that its median is `mean_reflectance`."""
    with rasterio.open(naip_path) as ds:
        rgb = ds.read([1, 2, 3]).astype(np.float64)
        bounds = ds.bounds
    lum = rgb.mean(axis=0)
    rho = np.clip(lum / np.median(lum[lum > 0]) * mean_reflectance, 0, 1)
    Image.fromarray(np.round(rho * 65535).astype(np.uint16)).save(out_path)
    return bounds


def solar_spectrum_xml(csv_path, scale, lo_nm, hi_nm, indent):
    """TSIS-1 HSRS (W/m²/nm), averaged to 1 nm bins, as a piecewise-linear spectrum in W/m²/µm times `scale`."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=(0, 1))
    nm = np.arange(lo_nm, hi_nm + 1)
    e = np.interp(nm, data[:, 0], data[:, 1]) * 1e3 * scale
    return "\n".join(f'{indent}<bin wavelength="{w}nm" value="{v:.6g}"/>' for w, v in zip(nm, e))


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
    p.add_argument("--reflectance", type=float, default=0.35, help="median reflectance of the placeholder map (default .35)")
    args = p.parse_args()

    site = json.loads((args.site_dir / "site.json").read_text())
    west, south, east, north = site["bounds"]
    centre_en = ((west + east) / 2, (south + north) / 2)
    centre_h = site["centre"]["elevation_m"]

    vertices, normals, E, N, faces = terrain(args.site_dir / site["dem"]["file"], args.mesh_spacing, centre_en, centre_h)
    tex_bounds = reflectance_map(args.site_dir / site["naip"]["file"], args.site_dir / "reflectance.png", args.reflectance)
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

    xml = f'''<?xml version="1.0" encoding="utf-8"?>
<!-- Generated by tools/remote_sensing/build_scene.py from {args.site_dir.name}: {site["size_m"]:g} m around
     {site["centre"]["lat"]}, {site["centre"]["lon"]} ({site["dem"]["source"]}, {site["dem"]["project"]}; NAIP {", ".join(site["naip"]["dates"])}).
     Frame: x east, y up, z south, metres, origin on the ground at the centre ({centre_h:.2f} m NAVD88).
     Camera: {args.altitude / 1e3:g} km, view zenith {args.view_zenith:g}°, azimuth {args.view_azimuth:g}°, {args.gsd:g} m GSD at nadir.
     Pixels record band-averaged radiance, W/m²/sr/µm, at the top of a (missing) atmosphere.
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
{band_xml}
			</response>
		</film>
	</sensor>

	<!-- the sun: TSIS-1 HSRS (W/m²/µm), at {d:.5f} AU -->
	<emitter type="directional">
		<transform name="to_world">
			<lookat origin="0m, 0m, 0m" target="{-to_sun[0]:.9f}m, {-to_sun[1]:.9f}m, {-to_sun[2]:.9f}m" up="0, 1, 0"/>
		</transform>
		<spectrum name="irradiance" type="piecewise_linear">
{solar_spectrum_xml(args.site_dir / site["solar_spectrum"]["file"], 1 / d**2, lo_nm, hi_nm, "			")}
		</spectrum>
	</emitter>

	<shape type="ply">
		<path value="terrain.ply"/>
		<quantity name="scale" value="1m"/>
		<bsdf type="diffuse">
			<texture name="reflectance" type="bitmap">
				<path value="reflectance.png"/>
				<string name="colour_encoding" value="linear"/>
				<string name="filter_type" value="bilinear"/>
				<string name="wrap_mode" value="clamp"/>
			</texture>
		</bsdf>
	</shape>
</scene>
'''
    (args.site_dir / f"{args.name}.xml").write_text(xml)
    print(f"wrote {args.site_dir}/terrain.ply, reflectance.png, {args.name}.xml; camera focal length {focal_length:g} m")


if __name__ == "__main__":
    main()
