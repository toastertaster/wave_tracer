# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "rasterio", "OpenEXR", "requests"]
# ///
"""Maps the materials of a site fetched by fetch_site.py, as abundances of measured reflectance spectra.

Unmixes NAIP into a linear mixture of materials (endmembers) of known reflectance, each pixel's reflectance in NAIP's
bands being a non-negative combination of theirs. Writes into the site directory:
  materials.exr   the materials' abundances, in channels R, G, B and A (one per material, up to four; 0 beyond
                  them), on NAIP's grid
  materials.json  the materials, their reflectance spectra at 1 nm, and the abundances' means
build_scene.py then gives the terrain a spectral_mix reflectance: the abundances weighting the spectra.

The materials' spectra come from the USGS Spectral Library Version 7 (Kokaly et al. 2017), convolved to an ASD
spectrometer's resolution (1 nm sampling, 350-2500 nm; downloaded once, 28 MB, into --library), or from a CSV file of
wavelengths in nm and reflectances. A material is NAME=SOURCE[*SCALE]: SOURCE a part of a library file name (which
must match one file) or a CSV path, SCALE a factor. The defaults are for a dry playa:
  playa    Stonewall Playa (Nevada) dry mud: an analogue of Railroad Valley Playa (whose own surface reflectance,
           measured by RadCalNet, can be given as a CSV instead)
  concrete light grey concrete road: bright grey surfaces (gravel roads, structures)
  asphalt  old asphalt road: dark surfaces
The first material is the background: NAIP is not radiometrically calibrated, so the median pixel is taken to be the
first material, and each band is scaled accordingly. Of NAIP's colour, only the detail finer than --colour-scale is
kept (the rest is the background's): NAIP's mosaics vary in colour over hundreds of metres (vignetting, seams between
flight lines). Brightness is kept at every scale. NAIP's 8-bit values are taken to be proportional to radiance.

NAIP's bands are taken to be boxes (those of the Leica ADS100 camera): blue 420-492, green 533-587, red 604-664, near
infrared 833-920 nm.

    uv run tools/remote_sensing/materials.py SITE_DIR [--material NAME=SOURCE[*SCALE] ...] [--colour-scale 256]
"""

import argparse
import fnmatch
import itertools
import json
import pathlib
import zipfile

import numpy as np
import OpenEXR
import rasterio
import requests
from scipy.ndimage import gaussian_filter

SPLIB07_ASD = ("https://www.sciencebase.gov/catalog/file/get/586e8c88e4b0f5ce109fccae"
               "?f=__disk__6f%2F81%2F62%2F6f8162500e6f3d250883a27b3def74cb77b927fc")
SPLIB07_ASD_FILE = "ASCIIdata_splib07b_cvASD.zip"
NAIP_BANDS = {"red": (604, 664), "green": (533, 587), "blue": (420, 492), "nir": (833, 920)}   # naip.tif's order
DEFAULT_MATERIALS = [
    "playa=Stonewall_Playa_Dry_Mud_2001",
    "concrete=Concrete_GDS375_Lt_Gry_Road",
    "asphalt=Asphalt_GDS376_Blck_Road_old",
]
NM = np.arange(350, 2501)       # the spectra's wavelengths, nm


def library_spectrum(library: pathlib.Path, pattern: str):
    """The library's spectrum whose file name contains `pattern` (a reflectance, AREF or RREF): at NM, with the
    library's missing values (deleted channels) interpolated."""
    with zipfile.ZipFile(library) as z:
        names = [n for n in z.namelist() if n.endswith("REF.txt") and "errorbars" not in n
                 and fnmatch.fnmatch(n.rsplit("/", 1)[-1], f"*{pattern}*")]
        if len(names) != 1:
            raise SystemExit(f"'{pattern}' matches {len(names)} spectra of {library.name}"
                             + (": " + ", ".join(n.rsplit("/", 1)[-1] for n in names[:10]) if names else ""))
        wl_name = next(n for n in z.namelist() if "Wavelengths_ASD" in n)
        wl = np.loadtxt(z.open(wl_name), skiprows=1) * 1e3
        r = np.loadtxt(z.open(names[0]), skiprows=1)
    ok = r > -1         # -1.23e34: deleted
    return np.interp(NM, wl[ok], r[ok]), names[0].rsplit("/", 1)[-1]


def csv_spectrum(path: pathlib.Path):
    data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    if data[0, 0] > data[-1, 0]:
        data = data[::-1]
    if NM[0] < data[0, 0] or NM[-1] > data[-1, 0]:
        print(f"{path}: {data[0, 0]:g}-{data[-1, 0]:g} nm; extended at its ends to {NM[0]}-{NM[-1]} nm")
    return np.interp(NM, data[:, 0], data[:, 1]), str(path)


def fetch_library(path: pathlib.Path):
    if path.exists():
        return
    print(f"downloading the USGS Spectral Library 7 (ASD resolution) into {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(SPLIB07_ASD, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = path.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.rename(path)


def band_means(spectrum):
    """A spectrum's means over NAIP's bands, in naip.tif's order."""
    return np.array([spectrum[(NM >= lo) & (NM <= hi)].mean() for lo, hi in NAIP_BANDS.values()])


def unmix(R, E):
    """Non-negative least squares of every pixel: the abundances a >= 0 minimizing |E a - R|. R: (bands, pixels),
    E: (bands, materials). Exact, by trying every set of materials (few) and keeping the best of those whose
    least-squares abundances are all non-negative."""
    n = E.shape[1]
    best = np.full(R.shape[1], np.inf)
    a = np.zeros((n, R.shape[1]))
    for k in range(1, n + 1):
        for subset in itertools.combinations(range(n), k):
            Es = E[:, subset]
            x = np.linalg.pinv(Es) @ R
            residual = ((Es @ x - R) ** 2).sum(axis=0)
            better = (x >= 0).all(axis=0) & (residual < best)
            best[better] = residual[better]
            a[:, better] = 0
            a[np.ix_(subset, better.nonzero()[0])] = x[:, better]
    return a, np.sqrt(best / R.shape[0])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("site_dir", type=pathlib.Path)
    p.add_argument("--material", action="append", metavar="NAME=SOURCE[*SCALE]",
                   help=f"a material (up to four; the first is the background). Default: {' '.join(DEFAULT_MATERIALS)}")
    p.add_argument("--library", type=pathlib.Path, default=pathlib.Path.home() / ".cache/wave_tracer" / SPLIB07_ASD_FILE,
                   help="the USGS Spectral Library 7 (ASD resolution) zip; downloaded if missing")
    p.add_argument("--colour-scale", type=float, default=256,
                   help="m: NAIP's colour varies only on finer scales than this (default 256; 0: NAIP's colour is "
                        "kept at every scale)")
    args = p.parse_args()

    materials = args.material or DEFAULT_MATERIALS
    if len(materials) > 4:
        raise SystemExit("at most four materials")
    names, spectra, sources = [], [], []
    for m in materials:
        name, _, source = m.partition("=")
        source, _, scale = source.partition("*")
        if not name or not source:
            raise SystemExit(f"'{m}': expected NAME=SOURCE[*SCALE]")
        if source.lower().endswith(".csv"):
            s, origin = csv_spectrum(pathlib.Path(source))
        else:
            fetch_library(args.library)
            s, origin = library_spectrum(args.library, source)
        names.append(name)
        spectra.append(s * float(scale or 1))
        sources.append({"name": name, "source": origin, "scale": float(scale or 1)})
    E = np.stack([band_means(s) for s in spectra], axis=1)       # (bands, materials)
    for name, e in zip(names, E.T):
        print(f"{name}: NAIP band reflectances " + ", ".join(f"{b} {v:.3f}" for b, v in zip(NAIP_BANDS, e)))

    site = json.loads((args.site_dir / "site.json").read_text())
    with rasterio.open(args.site_dir / site["naip"]["file"]) as ds:
        dn = ds.read().astype(np.float64)
        res = ds.res[0]
    valid = (dn > 0).all(axis=0)
    # brightness (all scales), and colour: each band's share of the brightness
    brightness = dn.mean(axis=0)
    colour = np.where(valid, dn / np.maximum(brightness, 1e-9), 1)
    if args.colour_scale > 0:
        sigma = args.colour_scale / res / 2
        weight = gaussian_filter(valid.astype(np.float64), sigma)
        background = np.stack([gaussian_filter(c * valid, sigma) for c in colour]) / np.maximum(weight, 1e-9)
    else:
        background = np.median(colour[:, valid], axis=1)[:, None, None]
    # the median pixel is the background material
    R = E[:, :1, None] * (brightness / np.median(brightness[valid]))[None] * (colour / background)
    R[:, ~valid] = 0

    bands, h, w = R.shape
    abundances = np.zeros((len(names), h * w))
    rms = np.zeros(h * w)
    R = R.reshape(bands, -1)
    for start in range(0, h * w, 1 << 20):
        sl = slice(start, start + (1 << 20))
        abundances[:, sl], rms[sl] = unmix(R[:, sl], E)
    abundances = abundances.reshape(len(names), h, w)
    rms = rms.reshape(h, w)

    channels = {}
    for c, key in enumerate("RGBA"):
        a = abundances[c] if c < len(names) else np.zeros((h, w))
        channels[key] = OpenEXR.Channel(a.astype(np.float16))
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, channels) as f:
        f.write(str(args.site_dir / "materials.exr"))

    means = abundances.reshape(len(names), -1).mean(axis=1)
    out = {
        "weights": "materials.exr",
        "materials": sources,
        "mean_abundances": dict(zip(names, map(float, means))),
        "naip_bands_nm": NAIP_BANDS,
        "colour_scale_m": args.colour_scale,
        "residual_rms": float(rms[valid].mean()),
        "wavelength_nm": NM.tolist(),
        "reflectance": {name: [round(float(v), 5) for v in s] for name, s in zip(names, spectra)},
    }
    (args.site_dir / "materials.json").write_text(json.dumps(out, indent=1) + "\n")
    dominant = np.argmax(abundances, axis=0)
    print("dominant material: " + ", ".join(f"{n} {np.mean(dominant[valid] == i):.2%}" for i, n in enumerate(names)))
    print("mean abundances: " + ", ".join(f"{n} {m:.3f}" for n, m in zip(names, means))
          + f"; residual {out['residual_rms']:.4f} rms")
    print(f"wrote {args.site_dir}/materials.exr, materials.json")


if __name__ == "__main__":
    main()
