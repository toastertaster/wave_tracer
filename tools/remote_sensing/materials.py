# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "rasterio", "OpenEXR", "requests", "pillow"]
# ///
"""Maps the materials of a site fetched by fetch_site.py, as abundances of measured reflectance spectra.

Two modes: one set of materials for the whole site (a natural site such as a playa), or, for sites with a lidar
surface model (fetch_site.py --lidar; e.g. towns), surface classes, each with its own materials (see "Classes" below).

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

Classes (--classes; the default when the site has a lidar surface model, which it needs, and Sentinel-2's surface
reflectance, sentinel2.tif):
  1. NAIP is calibrated to surface reflectance: per band, a robust linear fit of its means over Sentinel-2's 10 m
     pixels (clear land: SCL 4 and 5) to Sentinel-2's Level-2A reflectance (bands 2, 3, 4, 8 for NAIP's blue, green,
     red, near infrared).
  2. Each NAIP pixel gets a class: water (lidar class 9; or not high and dark in the near infrared, either without a
     lidar return or near water returns and at their height, since a collection can leave many of its water returns
     unclassified), building
     (lidar class 6, or at least --min-height above the bare earth and not vegetated), tree (as high, and vegetated),
     grass (lower, vegetated) or ground (the rest: roads, pavement, bare soil). Vegetated: NDVI >= --ndvi.
  3. NAIP's own shadows are found: the sun's position during NAIP's flight is fitted (the direction whose shadows,
     cast by the surface model, best match NAIP's dark pixels), and the shadowed pixels that are dark take the
     reflectance of the nearest sunlit pixel of their class. (The renderer casts its own shadows.)
  4. Each class's pixels are unmixed into its materials: of its candidates (CLASS_CANDIDATES, from the USGS library;
     --class-material CLASS:NAME=SOURCE[*SCALE] replaces a class's), the four whose mixtures best fit its sunlit
     pixels. Here the abundances are fractions of the surface: they sum to at most one, the rest being dark (shade,
     at scales finer than NAIP's pixels), so that each material keeps its own brightness.
  5. Walls, which NAIP does not see, get one material (--wall-material; default a light grey building brick):
     build_scene.py gives it to the surface model's steep faces that are not vegetation.
Writes classes.png (the classes, uint8, on NAIP's grid: CLASSES' order from 1), materials_<class>.exr (each class's
abundances; beyond the class, those of its nearest pixel, so that filtering does not darken its edges) and
materials.json.

    uv run tools/remote_sensing/materials.py SITE_DIR [--material NAME=SOURCE[*SCALE] ...] [--colour-scale 256]
    uv run tools/remote_sensing/materials.py SITE_DIR --classes [--class-material CLASS:NAME=SOURCE ...]
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
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import ndimage
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
CLASSES = ["ground", "grass", "tree", "building", "water"]     # classes.png: 1, 2, ... (0: no data)
# each class's candidate materials (USGS library), of which the (up to) four that best fit the class's pixels are used
CLASS_CANDIDATES = {
    "ground": ["asphalt=Asphalt_GDS376_Blck_Road_old", "concrete=Concrete_GDS375_Lt_Gry_Road",
               "soil=Stonewall_Playa_Dry_Mud_2001", "lawn=Lawn_Grass_GDS91_green",
               "dark_paving=Brick_GDS355_Paving_Dk_Gry", "tan_paving=Brick_GDS347_Paving_Tan",
               "dry_grass=Grass_Golden_Dry_GDS480", "sand=Sand_GrndIsle1_no_oil"],
    "grass": ["lawn=Lawn_Grass_GDS91_green", "dry_grass=Grass_Golden_Dry_GDS480",
              "soil=Stonewall_Playa_Dry_Mud_2001", "asphalt=Asphalt_GDS376_Blck_Road_old",
              "mixed_grass=Grass_dry.5+.5green_AMX28", "cheatgrass=Cheatgrass_ANPC1_field_calib"],
    "tree": ["maple=Maple_Leaves_DW92-1", "walnut=Walnut_Leaf_SUN", "oak=Oak_Oak-Leaf-1_fresh",
             "aspen=Aspen_Leaf-A_DW92-2", "dry_grass=Grass_Golden_Dry_GDS480", "asphalt=Asphalt_GDS376_Blck_Road_old"],
    "building": ["white_roof=Fiberglass_GDS335_Wh_Roofing", "grey_shingle=Asphalt_Shingle_GDS368_Lgray",
                 "dark_shingle=Asphalt_Shingle_GDS367_DkGry", "red_brick=Brick_GDS353_Building_MedRed",
                 "concrete=Concrete_GDS375_Lt_Gry_Road", "grey_brick=Brick_GDS354_Building_Lt_Gry",
                 "tan_shingle=Asphalt_Shingle_GDS366_Tan", "galvanized=GalvanizedSheetMetal_GDS334",
                 "tar_roof=Asphalt_Tar_GDS346_Blck_Roof", "green_roofing=Fiberglass_GDS336_Grn_Roofng"],
    "water": ["water=Seawater_Coast_Chl_SW1", "turbid_water=Water+Montmor_SWy-2+0.50g-l",
              "silty_water=Water+Montmor_SWy-2+5.01g-l", "marsh_water=MarshWater_CRMS121v69-NoGlnt"],
}
DEFAULT_WALL_MATERIAL = "facade=Brick_GDS354_Building_Lt_Gry"
S2_TO_NAIP = [2, 1, 0, 3]       # sentinel2.tif's bands (blue, green, red, nir) in naip.tif's order


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


def unmix(R, E, max_sum=None):
    """Non-negative least squares of every pixel: the abundances a >= 0 minimizing |E a - R|, R: (bands, pixels),
    E: (bands, materials); with `max_sum`, also sum(a) <= max_sum (fractions of the surface, the rest dark: shade).
    Exact, by trying every set of materials (few), each with the sum free and at max_sum, and keeping the best
    solution that satisfies the constraints."""
    n = E.shape[1]
    best = np.full(R.shape[1], np.inf)
    a = np.zeros((n, R.shape[1]))

    def consider(subset, x, Es):
        residual = ((Es @ x - R) ** 2).sum(axis=0)
        ok = (x >= -1e-12).all(axis=0) & (residual < best)
        if max_sum is not None:
            ok &= x.sum(axis=0) <= max_sum + 1e-9
        best[ok] = residual[ok]
        a[:, ok] = 0
        a[np.ix_(subset, ok.nonzero()[0])] = np.maximum(x[:, ok], 0)

    for k in range(1, n + 1):
        for subset in itertools.combinations(range(n), k):
            Es = E[:, subset]
            x = np.linalg.pinv(Es) @ R
            consider(subset, x, Es)
            if max_sum is not None:
                # the least squares with sum(x) = max_sum: x + P⁻¹1 (max_sum - 1ᵀx) / 1ᵀP⁻¹1, P = EᵀE
                q = np.linalg.pinv(Es.T @ Es) @ np.ones(k)
                consider(subset, x + np.outer(q, (max_sum - x.sum(axis=0)) / q.sum()), Es)
    return a, np.sqrt(best / R.shape[0])


def load_materials(specs, library):
    """Names, spectra (at NM) and sources of materials given as NAME=SOURCE[*SCALE]."""
    names, spectra, sources = [], [], []
    for m in specs:
        name, _, source = m.partition("=")
        source, _, scale = source.partition("*")
        if not name or not source:
            raise SystemExit(f"'{m}': expected NAME=SOURCE[*SCALE]")
        if source.lower().endswith(".csv"):
            s, origin = csv_spectrum(pathlib.Path(source))
        else:
            fetch_library(library)
            s, origin = library_spectrum(library, source)
        names.append(name)
        spectra.append(s * float(scale or 1))
        sources.append({"name": name, "source": origin, "scale": float(scale or 1)})
    return names, spectra, sources


def write_weights(path, abundances):
    """Up to four abundance maps as a half-float RGBA EXR."""
    h, w = abundances.shape[1:]
    channels = {key: OpenEXR.Channel((abundances[c] if c < len(abundances) else np.zeros((h, w))).astype(np.float16))
                for c, key in enumerate("RGBA")}
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, channels) as f:
        f.write(str(path))


def unmix_image(R, E, max_sum=None):
    """unmix() of an image R (bands, h, w), in blocks: the abundances (materials, h, w) and residuals (h, w)."""
    bands, h, w = R.shape
    R = R.reshape(bands, -1)
    a = np.zeros((E.shape[1], h * w), np.float32)
    rms = np.zeros(h * w, np.float32)
    for start in range(0, h * w, 1 << 20):
        sl = slice(start, start + (1 << 20))
        a[:, sl], rms[sl] = unmix(R[:, sl].astype(np.float64), E, max_sum)
    return a.reshape(-1, h, w), rms.reshape(h, w)


# --- classes ------------------------------------------------------------------------------------------------------

def choose_materials(R, E, most=4, max_sum=1.0):
    """Of candidate materials E (bands, candidates), the (up to) `most` whose mixtures (see unmix) best fit pixels R
    (bands, pixels): the least mean residual, preferring earlier candidates unless a later set fits 2% better.
    Returns their indices."""
    best, chosen = np.inf, None
    for subset in itertools.combinations(range(E.shape[1]), min(most, E.shape[1])):
        _, rms = unmix(R.astype(np.float64), E[:, subset], max_sum)
        if rms.mean() < 0.98 * best:
            best, chosen = rms.mean(), subset
    return list(chosen)


def calibrate(dn, naip_transform, s2, s2_transform, crs):
    """Gains and offsets that make NAIP's digital numbers (bands, h, w; 0: no data) Sentinel-2's surface
    reflectance, per band: robust (3 MAD) linear fits of NAIP's means over Sentinel-2's clear land pixels."""
    hs, ws = s2.shape[1:]
    fits = []
    clear = np.isin(s2[4], (4, 5))
    for b in range(4):
        mean = np.full((hs, ws), np.nan, np.float32)
        reproject(dn[b].astype(np.float32), mean, src_transform=naip_transform, src_crs=crs, dst_transform=s2_transform,
                  dst_crs=crs, resampling=Resampling.average, src_nodata=0, dst_nodata=np.nan)
        y = s2[S2_TO_NAIP[b]]
        ok = clear & np.isfinite(mean) & (y > 0)
        x, y = mean[ok].astype(np.float64), y[ok].astype(np.float64)
        inliers = np.ones(len(x), bool)
        for _ in range(8):
            A = np.stack([x[inliers], np.ones(inliers.sum())], 1)
            (g, o), *_ = np.linalg.lstsq(A, y[inliers], rcond=None)
            r = y - (g * x + o)
            mad = np.median(np.abs(r[inliers]))
            inliers = np.abs(r) < 3 * 1.4826 * mad
        r2 = 1 - np.var(r[inliers]) / np.var(y[inliers])
        fits.append((float(g), float(o), float(r2), int(inliers.sum())))
    return fits


def cast_shadows(height, res, azimuth, elevation, max_distance):
    """Cells of a height field (rows southwards) in the shadow of others, for the sun at `azimuth` (clockwise from
    north) and `elevation`, degrees: marching towards the sun, up to `max_distance` m."""
    az, el = np.radians(azimuth), np.radians(elevation)
    h, w = height.shape
    shadow = np.zeros((h, w), bool)
    seen = set()
    for step in range(1, int(max_distance / res) + 1):
        dc, dr = int(round(step * np.sin(az))), int(round(-step * np.cos(az)))
        if (dr, dc) in seen or abs(dr) >= h or abs(dc) >= w:
            continue
        seen.add((dr, dc))
        rise = np.hypot(dr, dc) * res * np.tan(el)
        # the height at (r + dr, c + dc), -inf beyond the edges
        shifted = np.full((h, w), -np.inf)
        r0, r1 = max(0, -dr), min(h, h - dr)
        c0, c1 = max(0, -dc), min(w, w - dc)
        shifted[r0:r1, c0:c1] = height[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        shadow |= shifted - rise > height + 0.5
    return shadow


def fit_sun(height, brightness, ground, res):
    """The sun's azimuth and elevation (degrees) whose shadows on `ground` cells best correlate with darkness."""
    span = float(np.percentile(height, 99.9) - np.percentile(height, 0.1))

    def score(az, el):
        s = cast_shadows(height, res, az, el, min(span / np.tan(np.radians(el)), 600))[ground]
        if s.mean() < 0.002:
            return -1.0
        return float(np.corrcoef(s, -brightness[ground])[0, 1])

    best = max(((score(az, el), az, el) for az in range(90, 271, 10) for el in range(25, 76, 10)))
    _, az0, el0 = best
    best = max(((score(az, el), az, el) for az in np.arange(az0 - 8, az0 + 9, 2) for el in np.arange(el0 - 8, el0 + 9, 2)
                if 10 <= el <= 85), default=best)
    return best


def main_classes(args, site):
    if not site.get("lidar"):
        raise SystemExit("--classes needs the site's lidar surface model: fetch_site.py --lidar")
    if not site.get("sentinel2"):
        raise SystemExit("--classes needs Sentinel-2's surface reflectance, to calibrate NAIP: fetch_site.py --sentinel2")
    d = args.site_dir
    with rasterio.open(d / site["naip"]["file"]) as ds:
        dn = ds.read().astype(np.float32)
        naip_transform, crs, res = ds.transform, ds.crs, ds.res[0]
    with rasterio.open(d / site["sentinel2"]["file"]) as ds:
        s2, s2_transform = ds.read(), ds.transform
    with rasterio.open(d / site["lidar"]["dsm"]) as ds:
        dsm = ds.read(1).astype(np.float64)
        lidar_res = ds.res[0]
    with rasterio.open(d / site["dem"]["file"]) as ds:
        dem = ds.read(1).astype(np.float64)
    with rasterio.open(d / site["lidar"]["classes"]) as ds:
        lidar_cls = ds.read(1)
    valid = (dn > 0).all(axis=0)
    h, w = valid.shape

    # 1. calibration
    fits = calibrate(dn, naip_transform, s2, s2_transform, crs)
    R = np.empty_like(dn)
    for b, (g, o, r2, n) in enumerate(fits):
        R[b] = np.clip(g * dn[b] + o, 0, 1)
        print(f"calibration, {list(NAIP_BANDS)[b]}: reflectance = {g:.5f} DN {o:+.4f} (r² {r2:.3f}, {n} pixels)")
    R[:, ~valid] = 0

    # 2. classes, on NAIP's grid
    rows = np.minimum(((np.arange(h) + 0.5) * res / lidar_res).astype(int), dsm.shape[0] - 1)
    cols = np.minimum(((np.arange(w) + 0.5) * res / lidar_res).astype(int), dsm.shape[1] - 1)
    hag = (dsm - dem)[np.ix_(rows, cols)]
    lc = lidar_cls[np.ix_(rows, cols)]
    red, nir = R[0], R[3]
    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    vegetated = ndvi >= args.ndvi
    high = hag >= args.min_height
    cls = np.full((h, w), CLASSES.index("ground") + 1, np.uint8)
    cls[vegetated] = CLASSES.index("grass") + 1
    cls[high & vegetated] = CLASSES.index("tree") + 1
    cls[(high & ~vegetated) | ((lc == 6) & ~vegetated)] = CLASSES.index("building") + 1
    # unclassified water: lidar can leave most of a river's returns unclassified, so water is also
    # what lies within 300 m of returns classified as water and within 1 m of their height, if dark (calibrated, water's
    # near infrared can be ~.07, as shaded pavement's; sunlit pavement's is ~.15). Elsewhere, only the darkest cells
    # without returns are.
    water = (lc == 0) & (nir < 0.06)
    if (lidar_cls == 9).any():
        dist, (ir, ic) = ndimage.distance_transform_edt(lidar_cls != 9, return_indices=True)
        level = dsm[ir, ic]
        near = (dist * lidar_res <= 300) & (np.abs(dsm - level) <= 1)
        water |= near[np.ix_(rows, cols)] & (nir < 0.12)
    water &= ~high
    cls[(lc == 9) | water] = CLASSES.index("water") + 1
    cls[~valid] = 0

    # 3. NAIP's shadows: fit the sun on a 2 m grid, cast its shadows on the lidar grid
    k = max(1, int(round(2 / lidar_res)))
    coarse = dsm[:dsm.shape[0] // k * k, :dsm.shape[1] // k * k].reshape(dsm.shape[0] // k, k, -1, k).max(axis=(1, 3))
    hag_c = (dsm - dem)[:coarse.shape[0] * k, :coarse.shape[1] * k].reshape(coarse.shape[0], k, -1, k).max(axis=(1, 3))
    bright = np.full(coarse.shape, np.nan)
    reproject(R[:3].mean(axis=0), bright, src_transform=naip_transform, src_crs=crs,
              dst_transform=rasterio.Affine(lidar_res * k, 0, naip_transform.c, 0, -lidar_res * k, naip_transform.f),
              dst_crs=crs, resampling=Resampling.average)
    ground_c = (hag_c < 1) & np.isfinite(bright)
    corr, az, el = fit_sun(coarse, np.nan_to_num(bright), ground_c, lidar_res * k)
    span = float(np.percentile(dsm, 99.9) - np.percentile(dsm, 0.1))
    shadow = cast_shadows(dsm, lidar_res, az, el, min(span / np.tan(np.radians(el)), 600))
    shadow = ndimage.binary_dilation(shadow, iterations=1)[np.ix_(rows, cols)]
    brightness = R[:3].mean(axis=0)
    dark = np.zeros((h, w), bool)
    for c in range(1, len(CLASSES) + 1):
        lit = (cls == c) & ~shadow
        if lit.any():
            dark |= (cls == c) & (brightness < 0.6 * np.median(brightness[lit]))
    shaded = shadow & dark & valid & (cls != CLASSES.index("water") + 1)
    print(f"NAIP's sun: azimuth {az:.0f}°, elevation {el:.0f}° (correlation {corr:.2f}); {shaded.mean():.1%} of the "
          "pixels shaded, filled from sunlit ones")

    # 4. unmix each class
    out_classes, mean_reflectance = [], np.zeros(len(NM))
    for c, cname in enumerate(CLASSES, start=1):
        member = cls == c
        fraction = float(member.mean())
        if fraction == 0:
            continue
        specs = args.class_materials.get(cname, CLASS_CANDIDATES[cname])
        names, spectra, sources = load_materials(specs, args.library)
        E = np.stack([band_means(s) for s in spectra], axis=1)
        # the class's sunlit pixels; every other pixel takes the nearest one's reflectance
        source = member & ~shaded
        if not source.any():
            source = member
        if len(names) > 4:
            pixels = np.flatnonzero(source)
            pixels = np.random.default_rng(0).choice(pixels, min(len(pixels), 20000), replace=False)
            keep = choose_materials(R.reshape(4, -1)[:, pixels], E)
            names, spectra, sources = [names[i] for i in keep], [spectra[i] for i in keep], [sources[i] for i in keep]
            E = E[:, keep]
        _, (ri, ci) = ndimage.distance_transform_edt(~source, return_indices=True)
        Rc = R[:, ri, ci]
        del ri, ci
        a, rms = unmix_image(Rc, E, max_sum=1.0)
        write_weights(d / f"materials_{cname}.exr", a)
        means = a[:, member].mean(axis=1)
        mean_reflectance += fraction * sum(m * s for m, s in zip(means, spectra))
        out_classes.append({
            "name": cname, "id": c, "fraction": fraction, "weights": f"materials_{cname}.exr",
            "materials": sources, "mean_abundances": dict(zip(names, map(float, means))),
            "residual_rms": float(rms[member].mean()),
            "reflectance": {n: [round(float(v), 5) for v in s] for n, s in zip(names, spectra)},
        })
        print(f"{cname}: {fraction:.1%} of the site; mean abundances "
              + ", ".join(f"{n} {m:.3f}" for n, m in zip(names, means)) + f"; residual {rms[member].mean():.4f} rms")
    Image.fromarray(cls).save(d / "classes.png")
    # walls, which NAIP does not see: one material
    names, spectra, sources = load_materials([args.wall_material], args.library)
    wall = {"name": "wall", "id": None, "fraction": 0.0, "weights": None, "materials": sources,
            "mean_abundances": {names[0]: 1.0}, "residual_rms": None,
            "reflectance": {names[0]: [round(float(v), 5) for v in spectra[0]]}}
    out = {
        "class_map": "classes.png",
        "classes": out_classes,
        "wall": wall,
        "calibration": {"source": site["sentinel2"]["item"],
                        "bands": {b: {"gain": g, "offset": o, "r2": r2, "pixels": n} for b, (g, o, r2, n) in zip(NAIP_BANDS, fits)}},
        "naip_sun": {"azimuth_deg": float(az), "elevation_deg": float(el), "correlation": corr,
                     "shaded_fraction": float(shaded.mean())},
        "ndvi_threshold": args.ndvi, "min_height_m": args.min_height,
        "naip_bands_nm": NAIP_BANDS,
        "wavelength_nm": NM.tolist(),
        "mean_reflectance": [round(float(v), 5) for v in mean_reflectance],
    }
    (d / "materials.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {d}/classes.png, " + ", ".join(c["weights"] for c in out_classes) + ", materials.json")


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
    p.add_argument("--classes", action=argparse.BooleanOptionalAction,
                   help="map surface classes, each with its own materials (default: if the site has lidar)")
    p.add_argument("--class-material", action="append", default=[], metavar="CLASS:NAME=SOURCE[*SCALE]",
                   help="a candidate material of a class (replaces the class's defaults; of more than four, the four "
                        "that best fit the class's pixels are used)")
    p.add_argument("--wall-material", default=DEFAULT_WALL_MATERIAL, metavar="NAME=SOURCE[*SCALE]",
                   help=f"classes: the material of walls (default {DEFAULT_WALL_MATERIAL})")
    p.add_argument("--ndvi", type=float, default=0.3, help="classes: NDVI of vegetation (default .3)")
    p.add_argument("--min-height", type=float, default=2.5,
                   help="classes: m above the bare earth of buildings and trees (default 2.5)")
    args = p.parse_args()

    site = json.loads((args.site_dir / "site.json").read_text())
    if args.classes or (args.classes is None and site.get("lidar")):
        args.class_materials = {}
        for spec in args.class_material:
            cname, _, m = spec.partition(":")
            if cname not in CLASSES:
                raise SystemExit(f"'{spec}': the classes are {', '.join(CLASSES)}")
            args.class_materials.setdefault(cname, []).append(m)
        return main_classes(args, site)

    materials = args.material or DEFAULT_MATERIALS
    if len(materials) > 4:
        raise SystemExit("at most four materials")
    names, spectra, sources = load_materials(materials, args.library)
    E = np.stack([band_means(s) for s in spectra], axis=1)       # (bands, materials)
    for name, e in zip(names, E.T):
        print(f"{name}: NAIP band reflectances " + ", ".join(f"{b} {v:.3f}" for b, v in zip(NAIP_BANDS, e)))

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

    write_weights(args.site_dir / "materials.exr", abundances)

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
