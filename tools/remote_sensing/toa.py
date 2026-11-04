# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "OpenEXR"]
# ///
"""Adds the atmosphere's own radiance to a render of a scene that build_scene.py built with an atmosphere.

The render records the radiance at the top of the atmosphere that comes straight from the ground (lit by the sun
and the sky, seen through the direct transmittance). This adds, per band (as 6S does):
  - the path radiance: sunlight that the atmosphere scatters towards the camera without reaching the ground;
  - the adjacency effect: light from the ground that the atmosphere scatters into the view. It is the render
    averaged with 6S's environment function (Tanré et al. 1981; a satellite's, corrected for the view zenith), whose
    reach is of kilometres; beyond the image the ground is taken to be as bright as the image's mean. Times the
    ratio of diffuse to direct upward transmittance, it is the radiance that the diffuse transmittance brings.
The image is taken to be a grid on the ground of the nadir sampling distance; off nadir that is an approximation.
A render through a telescope's mirrors (build_scene.py --telescope) has their reflectance in it: the path radiance is
then multiplied by scene.json's telescope band_throughput (the mirrors' reflectance at normal incidence, averaged over
the band), so that OUT is the radiance at the focal plane, as the render is. (Stray light from the atmosphere's own
radiance outside the field is not added.)

    uv run tools/remote_sensing/toa.py SITE_DIR/scene.json RENDER.exr OUT.exr [--components]

SITE_DIR/scene.json is written by build_scene.py with the scene. OUT.exr has the render's channels (one per band,
named after it), in the render's units (W/m²/sr/µm, band-averaged).
"""

import argparse
import json
import math
import pathlib

import numpy as np
import OpenEXR
from scipy.signal import fftconvolve


def environment_function(r_km, view_zenith_deg, rayleigh_fraction):
    """6S's environment function for a satellite (ENVIRO.f): the fraction of the diffusely transmitted radiance
    that comes from the ground within r_km of the viewed point, molecules' and aerosols' weighted by
    `rayleigh_fraction`; and its derivative, per km."""
    er1, er2 = np.exp(-0.080 * r_km), np.exp(-1.100 * r_km)
    ea1, ea2 = np.exp(-0.270 * r_km), np.exp(-2.830 * r_km)
    fra0, dfra0 = 1 - 0.930 * er1 - 0.070 * er2, 0.930 * 0.080 * er1 + 0.070 * 1.100 * er2
    fae0, dfae0 = 1 - 0.448 * ea1 - 0.552 * ea2, 0.448 * 0.270 * ea1 + 0.552 * 2.830 * ea2
    # correction for the view zenith angle: polynomials in fra0 and fae0
    a0, b0, a1, b1 = 1.3347, 0.57757, -1.479, -1.5275
    x = math.log(math.cos(math.radians(view_zenith_deg)))
    c0, c1, c2 = 1 + a0 * x + b0 * x * x, a1 * x + b1 * x * x, (-a1 - a0) * x + (-b1 - b0) * x * x
    fra, dfra = fra0 * (x * (1 - fra0) + 1), dfra0 * (x * (1 - 2 * fra0) + 1)
    fae, dfae = fae0 * (c0 + fae0 * (c1 + fae0 * c2)), dfae0 * (c0 + fae0 * (2 * c1 + 3 * fae0 * c2))
    w = rayleigh_fraction
    return w * fra + (1 - w) * fae, w * dfra + (1 - w) * dfae


def environment_kernel(height, width, gsd_m, view_zenith_deg, rayleigh_fraction):
    """The environment function's weights of the pixels at offsets up to the image's size: of each, F's density
    (dF/dr / 2πr) times its area, averaged over 8x8 samples near the centre; the centre's is F within the radius of a
    disc of a pixel's area."""
    def density(r_m):   # per m²
        dF = environment_function(r_m / 1e3, view_zenith_deg, rayleigh_fraction)[1] / 1e3
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r_m > 0, dF / (2 * math.pi * r_m), 0)

    y = np.arange(-(height - 1), height)[:, None] * gsd_m
    x = np.arange(-(width - 1), width)[None, :] * gsd_m
    K = density(np.hypot(y, x)) * gsd_m**2
    near = 4
    s = (np.arange(8) + 0.5) / 8 - 0.5
    for i in range(-near, near + 1):
        for j in range(-near, near + 1):
            if i or j:
                yy, xx = np.meshgrid((i + s) * gsd_m, (j + s) * gsd_m, indexing="ij")
                K[height - 1 + i, width - 1 + j] = density(np.hypot(yy, xx)).mean() * gsd_m**2
    K[height - 1, width - 1] = environment_function(1e-3 * gsd_m / math.sqrt(math.pi), view_zenith_deg,
                                                    rayleigh_fraction)[0]
    return K


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene_json", type=pathlib.Path)
    p.add_argument("render", type=pathlib.Path)
    p.add_argument("out", type=pathlib.Path)
    p.add_argument("--components", action="store_true",
                   help="also write OUT's path radiance and adjacency radiance, as OUT_path.exr and OUT_adjacency.exr")
    args = p.parse_args()

    scene = json.loads(args.scene_json.read_text())
    atm = scene["atmosphere"]
    if atm is None:
        raise SystemExit(f"{args.scene_json}: the scene has no atmosphere")

    with OpenEXR.File(str(args.render), separate_channels=True) as f:
        header = dict(f.header())
        channels = {name: np.asarray(ch.pixels, dtype=np.float64) for name, ch in f.channels().items()}
    missing = set(scene["bands_nm"]) - set(channels)
    if missing:
        raise SystemExit(f"{args.render} has channels {sorted(channels)}, not bands {sorted(missing)}")

    # a render through a telescope's mirrors (build_scene.py --telescope) has their reflectance in it, which the
    # path radiance must pass too (the adjacency radiance, from the render, has it already)
    throughput = (scene.get("telescope") or {}).get("band_throughput", {})
    out, path, adjacency = {}, {}, {}
    for band in scene["bands_nm"]:
        L = channels[band]
        K = environment_kernel(*L.shape, scene["gsd_m"], scene["view_zenith_deg"], atm["rayleigh_fraction"][band])
        mean = L.mean()
        averaged = mean + fftconvolve(L - mean, K, mode="same")
        adjacency[band] = atm["diffuse_to_direct_up"][band] * averaged
        path[band] = np.full_like(L, atm["path_radiance"][band] * throughput.get(band, 1.0))
        out[band] = L + adjacency[band] + path[band]
        through = f" × telescope {throughput[band]:.4g}" if band in throughput else ""
        print(f"{band}: render mean {mean:.4g}, + path {atm['path_radiance'][band]:.4g}{through}, "
              f"+ adjacency mean {adjacency[band].mean():.4g} = {out[band].mean():.4g} W/m²/sr/µm")

    def write(path_, images):
        hdr = {k: v for k, v in header.items() if k in ("renderer", "scene", "sensor", "samples")}
        chans = {k: OpenEXR.Channel(v.astype(np.float32)) for k, v in images.items()}
        with OpenEXR.File(hdr, chans) as f:
            f.write(str(path_))

    write(args.out, out)
    if args.components:
        write(args.out.with_name(args.out.stem + "_path.exr"), path)
        write(args.out.with_name(args.out.stem + "_adjacency.exr"), adjacency)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
