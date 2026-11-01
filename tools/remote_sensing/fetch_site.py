# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "rasterio", "pyproj", "pystac-client", "planetary-computer", "requests", "laspy[lazrs]", "shapely"]
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
and, optionally:
  dsm.tif, lidar_class.tif  (--lidar)
             a surface model from USGS 3DEP's lidar point clouds: the highest
             return of each cell (roofs, tree crowns, bridges; float32 metres,
             NAVD88, its datum matched to the DEM's), and its ASPRS class
             (uint8: 2 ground, 3-5 vegetation, 6 building, 9 water, 17 bridge;
             0 where no return fell, filled from the DEM). The point clouds
             (3DEP's Entwine Point Tiles on AWS, else its LAZ tiles; see
             lidar.py) are downloaded once into --lidar-cache.
  sentinel2.tif  (--sentinel2)
             Sentinel-2's Level-2A surface reflectance (blue, green, red, near
             infrared: bands 2, 3, 4, 8, float32; then the scene
             classification, SCL) of the clearest pass over the area within
             --sentinel2-days of --date, on a 10 m grid: materials.py
             calibrates NAIP with it
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
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lidar  # noqa: E402

TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"
DEM_DATASET = "Digital Elevation Model (DEM) 1 meter"
STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
TSIS_HSRS = "https://lasp.colorado.edu/lisird/latis/dap/tsis1_hsrs_1nm.csv"
AERONET_SITES = "https://aeronet.gsfc.nasa.gov/aeronet_locations_v3.txt"
AERONET_DATA = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"


def get(url, tries=4, **kwargs):
    """requests.get, retried on timeouts, connection errors and server errors (these services are often busy)."""
    timeout = kwargs.pop("timeout", 60)
    for attempt in range(tries):
        try:
            r = requests.get(url, timeout=timeout, **kwargs)
            if r.status_code < 500 or attempt == tries - 1:
                r.raise_for_status()
                return r
        except (requests.ConnectionError, requests.Timeout):
            if attempt == tries - 1:
                raise
        print(f"retrying {url}", file=sys.stderr)


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


def dem_projects(bbox):
    """The 1 m DEM projects with tiles in the bounding box, newest first: {project: tile URLs}."""
    params = {"datasets": DEM_DATASET, "bbox": ",".join(map(str, bbox)), "max": 200, "outputFormat": "JSON"}
    items = get(TNM_API, params=params).json()["items"]
    if not items:
        sys.exit("no 3DEP 1 m DEM covers this area")
    items.sort(key=lambda i: i["publicationDate"], reverse=True)
    projects = {}
    for i in items:
        # .../Elevation/1m/Projects/<project>/TIFF/<tile>.tif
        project = i["downloadURL"].split("/Projects/")[-1].split("/")[0]
        projects.setdefault(project, []).append(i["downloadURL"])
    return projects


def read_dem(bounds, crs: CRS, nodata=-999999.0):
    """The DEM over the area at 1 m, its transform, its project and its tiles' URLs: of the newest project that covers
    the whole area (tiles of different projects overlap and may disagree), else of all projects, the newer covering
    the older."""
    projects = dem_projects(lonlat_bbox(bounds, crs))
    for project, urls in projects.items():
        dem, transform, _ = merge(["/vsicurl/" + u for u in urls], bounds, 1.0, Resampling.bilinear, nodata)
        if not (dem == nodata).any():
            return dem, transform, project, urls
        print(f"DEM: {project} does not cover the whole area", file=sys.stderr)
    urls = [u for us in projects.values() for u in us]
    dem, transform, _ = merge(["/vsicurl/" + u for u in urls], bounds, 1.0, Resampling.bilinear, nodata)
    if (dem == nodata).any():
        sys.exit("the DEMs do not cover the whole area")
    return dem, transform, " + ".join(projects), urls


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
    lines = get(AERONET_SITES).text.splitlines()
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
        text = get(AERONET_DATA, params=params).text
        # (the service also returns neighbouring days)
        rows = [l for l in text.splitlines() if l.startswith(f"{name},{date:%d:%m:%Y},")]
        if rows:
            header = next(l for l in text.splitlines() if l.startswith("AERONET_Site,"))
            return name, d, header + "\n" + "\n".join(rows) + "\n"
    return None


S2_BANDS = ["B02", "B03", "B04", "B08"]      # blue, green, red, near infrared (10 m)
S2_CLOUDY = (3, 8, 9, 10)                   # SCL: cloud shadow, cloud (medium, high probability), cirrus


def read_warped(href, bounds, crs, res, resampling):
    """A band of a cloud-optimized GeoTIFF, warped onto the area's grid at `res`."""
    with rasterio.open(href) as src, WarpedVRT(src, crs=crs, resampling=resampling) as vrt:
        window = from_bounds(*bounds, transform=vrt.transform)
        w, h = int(round((bounds[2] - bounds[0]) / res)), int(round((bounds[3] - bounds[1]) / res))
        return vrt.read(1, window=window, out_shape=(h, w), resampling=resampling)


def sentinel2(bounds, crs: CRS, date: datetime.date, days: int):
    """The clearest Sentinel-2 L2A pass over the area within `days` of `date` (the nearest of those with < 1% of
    clouds over it, else the least cloudy): its surface reflectance in bands 2, 3, 4, 8 and its SCL, on a 10 m grid."""
    catalog = pystac_client.Client.open(STAC_API, modifier=planetary_computer.sign_inplace)
    span = f"{date - datetime.timedelta(days=days)}/{date + datetime.timedelta(days=days)}"
    items = list(catalog.search(collections=["sentinel-2-l2a"], bbox=lonlat_bbox(bounds, crs), datetime=span,
                                query={"eo:cloud_cover": {"lt": 50}}).items())
    items.sort(key=lambda i: (abs((i.datetime.date() - date).days), i.properties["eo:cloud_cover"]))
    best = None
    for item in items:
        scl = read_warped(item.assets["SCL"].href, bounds, crs, 10.0, Resampling.nearest)
        cloudy = float(np.isin(scl, S2_CLOUDY).mean() + (scl == 0).mean())      # 0: no data
        if best is None or cloudy < best[1]:
            best = (item, cloudy, scl)
        if cloudy < 0.01:
            break
    if best is None:
        return None
    item, cloudy, scl = best
    # baseline 04.00 and later (from 2022-01-25) add 1000 to the digital numbers
    offset = -1000 if float(item.properties.get("s2:processing_baseline", "0")) >= 4 else 0
    bands = [(read_warped(item.assets[b].href, bounds, crs, 10.0, Resampling.bilinear).astype(np.float32) + offset) / 1e4
             for b in S2_BANDS]
    return item, cloudy, np.stack(bands + [scl.astype(np.float32)])


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
    parser.add_argument("--lidar", action="store_true", help="also make a surface model from 3DEP's lidar")
    parser.add_argument("--lidar-project", help="a 3DEP collection to use (part of its name; default: the newest "
                        "that covers the area, from the Entwine Point Tiles if any)")
    parser.add_argument("--lidar-cache", type=pathlib.Path, default=pathlib.Path.home() / ".cache/wave_tracer/lidar",
                        help="where the point clouds are kept (default ~/.cache/wave_tracer/lidar)")
    parser.add_argument("--sentinel2", action="store_true", help="also fetch Sentinel-2's surface reflectance")
    parser.add_argument("--sentinel2-days", type=int, default=20, help="search window around --date (default 20)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    crs = utm_crs(args.lon)
    bounds = aoi_bounds(args.lat, args.lon, args.size, crs)
    print(f"area: {args.size:g} m square around {args.lat}, {args.lon}: {crs.name} {bounds}")

    dem, dem_transform, project, urls = read_dem(bounds, crs)
    print(f"DEM: 3DEP 1 m project {project}, {len(urls)} tile(s)")
    write(args.out_dir / "dem.tif", dem.astype(np.float32), dem_transform, crs)

    lidar_info = None
    if args.lidar:
        dsm, cls, lidar_info = lidar.fetch(bounds, crs, dem[0].astype(np.float64), args.lidar_cache, args.lidar_project)
        write(args.out_dir / "dsm.tif", dsm[None], dem_transform, crs)
        write(args.out_dir / "lidar_class.tif", cls[None], dem_transform, crs)
        classes = {int(c): float(np.mean(cls == c)) for c in np.unique(cls)}
        print(f"lidar: {lidar_info['returns']:,} returns ({lidar_info['returns'] / args.size**2:.1f}/m²); heights in "
              f"{'m' if lidar_info['height_unit_m'] == 1 else 'US survey ft'}, datum offset from the DEM "
              f"{lidar_info['datum_offset_m']:+.2f} m; {lidar_info['filled_from_dem']:.1%} of cells without returns "
              f"(from the DEM); {lidar_info['spikes_removed']:.3%} spikes removed; highest {np.max(dsm - dem[0]):.1f} m "
              "above ground; classes "
              + ", ".join(f"{c}: {f:.1%}" for c, f in classes.items()))
        lidar_info = {"dsm": "dsm.tif", "classes": "lidar_class.tif", "resolution_m": 1.0, **lidar_info}

    items = naip_items(bounds, crs, args.date)
    naip_res = min(i.properties["gsd"] for i in items)
    print(f"NAIP: {len(items)} item(s) of {items[0].datetime.year}, {naip_res} m")
    naip, naip_transform, _ = merge([i.assets["image"].href for i in items], bounds, naip_res, Resampling.bilinear, 0)
    if (naip[:3].max(axis=0) == 0).any():
        print("warning: the NAIP imagery leaves parts of the area empty", file=sys.stderr)
    write(args.out_dir / "naip.tif", naip, naip_transform, crs)

    solar = get(TSIS_HSRS, timeout=120)
    (args.out_dir / "tsis1_hsrs_1nm.csv").write_text(solar.text)
    print("solar spectrum: TSIS-1 HSRS (1 nm)")

    s2_info = None
    if args.sentinel2:
        found = sentinel2(bounds, crs, args.date, args.sentinel2_days)
        if found is None:
            print(f"Sentinel-2: no pass within {args.sentinel2_days} days of {args.date}", file=sys.stderr)
        else:
            item, cloudy, s2 = found
            transform = rasterio.transform.from_origin(bounds[0], bounds[3], 10.0, 10.0)
            write(args.out_dir / "sentinel2.tif", s2, transform, crs)
            print(f"Sentinel-2: {item.id}, {cloudy:.1%} of the area cloudy or empty")
            s2_info = {"file": "sentinel2.tif", "item": item.id, "datetime": item.datetime.isoformat(),
                       "bands": ["blue (B02)", "green (B03)", "red (B04)", "nir (B08)", "SCL"],
                       "cloudy_fraction": round(cloudy, 4), "source": "Sentinel-2 Level-2A (Planetary Computer)"}

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
        "lidar": lidar_info,
        "sentinel2": s2_info,
        "date": str(args.date),
        "aeronet": {"file": "aeronet.csv", "site": aeronet[0], "distance_km": round(aeronet[1], 2)} if aeronet else None,
        "solar_spectrum": {"file": "tsis1_hsrs_1nm.csv", "source": "TSIS-1 HSRS, 1 nm resolution", "url": TSIS_HSRS,
                           "units": "W/m^2/nm at 1 AU"},
    }
    (args.out_dir / "site.json").write_text(json.dumps(site, indent=2) + "\n")
    extra = ("dsm.tif, lidar_class.tif, " if lidar_info else "") + ("sentinel2.tif, " if s2_info else "")
    print(f"wrote {args.out_dir}/dem.tif, naip.tif, {extra}tsis1_hsrs_1nm.csv, site.json")


if __name__ == "__main__":
    main()
