# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "rasterio", "pyproj", "pystac-client", "planetary-computer", "requests"]
# ///
"""Fetches the terrain and imagery of a remote sensing reference site.

Downloads, for a square area around a point in the conterminous US:
  dem.tif    USGS 3DEP 1 m bare-earth elevation (float32 metres, NAVD88)
  naip.tif   the NAIP orthophoto nearest a date (uint8 red, green, blue, near
             infrared; 0.6 m in recent years)
  tsis1_hsrs_1nm.csv
             the TSIS-1 Hybrid Solar Reference Spectrum (Coddington et al.
             2021): top-of-atmosphere solar spectral irradiance at 1 AU,
             W/m²/nm, 202-2730 nm at 1 nm resolution
  aeronet.csv
             AERONET's Level 2.0 daily averages of aerosol optical depth and
             precipitable water, for --date, from the nearest site within
             --aeronet-radius (if any): inputs for atmosphere.py
  site.json  the area, its coordinate system, and where the data came from
into OUT_DIR. Both rasters share the area's bounds in its UTM zone (NAD83),
north up. Only the area is read from the sources (they are cloud-optimized
GeoTIFFs), not whole tiles.

The defaults are the RadCalNet site at Railroad Valley Playa, Nevada, and the
reference date of docs/source/remote_sensing/data.rst. build_scene.py turns
the result into a wave_tracer scene.

    uv run tools/remote_sensing/fetch_site.py OUT_DIR [--lat 38.504 --lon -115.692 --size 2048]
"""

import argparse
import datetime
import json
import math
import pathlib
import sys

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import rasterio.merge
import requests
from pyproj import CRS, Transformer
from rasterio.enums import Resampling

TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"
DEM_DATASET = "Digital Elevation Model (DEM) 1 meter"
STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
TSIS_HSRS = "https://lasp.colorado.edu/lisird/latis/dap/tsis1_hsrs_1nm.csv"
AERONET_SITES = "https://aeronet.gsfc.nasa.gov/aeronet_locations_v3.txt"
AERONET_DATA = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"


def utm_crs(lon: float) -> CRS:
    """NAD83 / UTM, the coordinate system of 3DEP's 1 m DEMs and of NAIP."""
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg(26900 + zone)


def aoi_bounds(lat: float, lon: float, size: float, crs: CRS):
    """The square area of side `size` metres centred on the point, snapped to whole metres."""
    east, north = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    west, south = round(east - size / 2), round(north - size / 2)
    return (west, south, west + size, south + size)


def lonlat_bbox(bounds, crs: CRS):
    to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    west, south, east, north = bounds
    corners = [to_lonlat.transform(x, y) for x in (west, east) for y in (south, north)]
    lons, lats = zip(*corners)
    return (min(lons), min(lats), max(lons), max(lats))


def dem_tiles(bbox):
    """URLs of the 1 m DEM tiles that cover the bounding box, newest project first."""
    params = {"datasets": DEM_DATASET, "bbox": ",".join(map(str, bbox)), "max": 100, "outputFormat": "JSON"}
    items = requests.get(TNM_API, params=params, timeout=60).json()["items"]
    if not items:
        sys.exit("no 3DEP 1 m DEM covers this area")
    items.sort(key=lambda i: i["publicationDate"], reverse=True)
    # one project only: tiles of different projects overlap and may disagree
    project = items[0]["title"].split()[-1]
    return project, [i["downloadURL"] for i in items if i["title"].endswith(project)]


def naip_items(bounds, crs: CRS, date: datetime.date):
    """The NAIP items of the acquisition year nearest `date` that intersect the area."""
    catalog = pystac_client.Client.open(STAC_API, modifier=planetary_computer.sign_inplace)
    bbox = lonlat_bbox(bounds, crs)
    items = list(catalog.search(collections=["naip"], bbox=bbox).items())
    if not items:
        sys.exit("no NAIP imagery covers this area")
    def distance(item):
        return abs((item.datetime.date() - date).days)
    year = min(items, key=distance).datetime.year
    return [i for i in items if i.datetime.year == year]


def aeronet_daily(lat: float, lon: float, date: datetime.date, radius_km: float):
    """The nearest AERONET site with Level 2.0 data on `date` within `radius_km`, and its daily averages (CSV)."""
    lines = requests.get(AERONET_SITES, timeout=60).text.splitlines()
    sites = []
    for line in lines[2:]:
        name, slon, slat, *_ = line.split(",")
        d = 6371 * math.acos(min(1, math.sin(math.radians(lat)) * math.sin(math.radians(float(slat))) +
                                 math.cos(math.radians(lat)) * math.cos(math.radians(float(slat))) *
                                 math.cos(math.radians(lon - float(slon)))))
        if d <= radius_km:
            sites.append((d, name))
    for d, name in sorted(sites):
        params = {"site": name, "year": date.year, "month": date.month, "day": date.day, "year2": date.year,
                  "month2": date.month, "day2": date.day, "AOD20": 1, "AVG": 20, "if_no_html": 1}
        text = requests.get(AERONET_DATA, params=params, timeout=60).text
        # (the service also returns neighbouring days)
        rows = [l for l in text.splitlines() if l.startswith(f"{name},{date:%d:%m:%Y},")]
        if rows:
            header = next(l for l in text.splitlines() if l.startswith("AERONET_Site,"))
            return name, d, header + "\n" + "\n".join(rows) + "\n"
    return None


def merge(urls, bounds, res, resampling, nodata=None):
    sources = [rasterio.open(u) for u in urls]
    try:
        data, transform = rasterio.merge.merge(sources, bounds=bounds, res=res, resampling=resampling, nodata=nodata)
        return data, transform, sources[0].profile
    finally:
        for s in sources:
            s.close()


def write(path, data, transform, crs, nodata=None):
    profile = {"driver": "GTiff", "height": data.shape[1], "width": data.shape[2], "count": data.shape[0],
               "dtype": data.dtype, "crs": crs, "transform": transform, "compress": "deflate",
               "tiled": True, "blockxsize": 512, "blockysize": 512}
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as out:
        out.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--lat", type=float, default=38.504, help="centre latitude (default: Railroad Valley RadCalNet site)")
    parser.add_argument("--lon", type=float, default=-115.692, help="centre longitude")
    parser.add_argument("--size", type=float, default=2048, help="side of the square area, metres (default 2048)")
    parser.add_argument("--date", type=datetime.date.fromisoformat, default=datetime.date(2022, 7, 19),
                        help="choose the NAIP year nearest this date, and AERONET's data of it (default 2022-07-19)")
    parser.add_argument("--aeronet-radius", type=float, default=50, help="km (default 50)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    crs = utm_crs(args.lon)
    bounds = aoi_bounds(args.lat, args.lon, args.size, crs)
    print(f"area: {args.size:g} m square around {args.lat}, {args.lon}: {crs.name} {bounds}")

    project, urls = dem_tiles(lonlat_bbox(bounds, crs))
    print(f"DEM: 3DEP 1 m project {project}, {len(urls)} tile(s)")
    nodata = -999999.0
    dem, dem_transform, _ = merge(["/vsicurl/" + u for u in urls], bounds, 1.0, Resampling.bilinear, nodata)
    if (dem == nodata).any():
        sys.exit("the DEM does not cover the whole area")
    write(args.out_dir / "dem.tif", dem.astype(np.float32), dem_transform, crs)

    items = naip_items(bounds, crs, args.date)
    naip_res = min(i.properties["gsd"] for i in items)
    print(f"NAIP: {len(items)} item(s) of {items[0].datetime.year}, {naip_res} m")
    naip, naip_transform, _ = merge([i.assets["image"].href for i in items], bounds, naip_res, Resampling.bilinear, 0)
    if (naip[:3].max(axis=0) == 0).any():
        print("warning: the NAIP imagery leaves parts of the area empty", file=sys.stderr)
    write(args.out_dir / "naip.tif", naip, naip_transform, crs)

    solar = requests.get(TSIS_HSRS, timeout=120)
    solar.raise_for_status()
    (args.out_dir / "tsis1_hsrs_1nm.csv").write_text(solar.text)
    print("solar spectrum: TSIS-1 HSRS (1 nm)")

    aeronet = aeronet_daily(args.lat, args.lon, args.date, args.aeronet_radius)
    if aeronet:
        (args.out_dir / "aeronet.csv").write_text(aeronet[2])
        print(f"AERONET: {aeronet[0]} ({aeronet[1]:.1f} km), Level 2.0 daily average of {args.date}")
    else:
        print(f"AERONET: no Level 2.0 data of {args.date} within {args.aeronet_radius:g} km", file=sys.stderr)

    centre_elevation = float(dem[0, dem.shape[1] // 2, dem.shape[2] // 2])
    site = {
        "centre": {"lat": args.lat, "lon": args.lon, "elevation_m": centre_elevation},
        "crs": crs.to_string(),
        "bounds": list(bounds),
        "size_m": args.size,
        "dem": {"file": "dem.tif", "resolution_m": 1.0, "source": "USGS 3DEP 1 m DEM", "project": project,
                "tiles": urls, "vertical_datum": "NAVD88"},
        "naip": {"file": "naip.tif", "resolution_m": naip_res, "bands": ["red", "green", "blue", "nir"],
                 "items": [i.id for i in items], "dates": sorted({str(i.datetime.date()) for i in items})},
        "date": str(args.date),
        "aeronet": {"file": "aeronet.csv", "site": aeronet[0], "distance_km": round(aeronet[1], 2)} if aeronet else None,
        "solar_spectrum": {"file": "tsis1_hsrs_1nm.csv", "source": "TSIS-1 HSRS, 1 nm resolution", "url": TSIS_HSRS,
                           "units": "W/m^2/nm at 1 AU"},
    }
    (args.out_dir / "site.json").write_text(json.dumps(site, indent=2) + "\n")
    print(f"wrote {args.out_dir}/dem.tif, naip.tif, tsis1_hsrs_1nm.csv, site.json")


if __name__ == "__main__":
    main()
