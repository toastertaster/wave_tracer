"""USGS 3DEP lidar point clouds as surface models, for fetch_site.py.

Reads the returns that fall in an area from 3DEP's point clouds, and rasterizes them onto the area's grid: the
highest return of each cell (a digital surface model: roofs, tree crowns, bridges) and its ASPRS class (1
unclassified, 2 ground, 3-5 vegetation, 6 building, 9 water, 17 bridge deck). The point clouds come from

  - the Entwine Point Tiles of 3DEP on AWS (s3://usgs-lidar-public; Hobu, Inc. and USGS): octrees, of which only the
    nodes over the area are read (in parallel). Their index (resources.geojson) lists each collection's boundary;
  - else 3DEP's LAZ tiles (The National Map), whole tiles of hundreds of MB, from a server that throttles downloads.

Both are downloaded once into a cache. Needs numpy, scipy, pyproj, requests and laspy with a LAZ backend
(laspy[lazrs]); shapely for the EPT index.
"""

import concurrent.futures
import json
import pathlib
import re
import sys

import numpy as np
import requests
from pyproj import CRS, Transformer
from scipy import ndimage

TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"
LPC_DATASET = "Lidar Point Cloud (LPC)"
EPT_INDEX = "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson"
NOISE_CLASSES = (7, 18)     # low and high noise
NO_DATA_CLASS = 0           # cells without returns, filled from the DEM
# rockyweb throttles python-requests' default user agent
HEADERS = {"User-Agent": "wave_tracer-tools (tools/remote_sensing/fetch_site.py)"}


def collection_year(name: str) -> int:
    """A 3DEP collection's year, from its name (..._2017, ..._D23 for 2023), or 0."""
    years = re.findall(r"(?:^|_)(?:(20\d\d)|[A-D](\d\d))(?:_|$)", name)
    return max((int(a) if a else 2000 + int(b)) for a, b in years) if years else 0


def area_lonlat(bounds, crs: CRS):
    to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    west, south, east, north = bounds
    xs, ys = np.meshgrid(np.linspace(west, east, 9), np.linspace(south, north, 9))
    return to_lonlat.transform(xs.ravel(), ys.ravel())


# --- Entwine Point Tiles ------------------------------------------------------------------------------------------

def ept_collections(bounds, crs: CRS, cache: pathlib.Path):
    """The EPT collections whose boundaries contain the whole area, newest first: (name, ept.json URL, points)."""
    from shapely.geometry import MultiPoint, shape
    index = cache / "resources.geojson"
    if not index.exists():
        cache.mkdir(parents=True, exist_ok=True)
        r = requests.get(EPT_INDEX, headers=HEADERS, timeout=120)
        r.raise_for_status()
        index.write_text(r.text)
    lons, lats = area_lonlat(bounds, crs)
    area = MultiPoint(list(zip(lons, lats))).convex_hull
    found = [(f["properties"]["name"], f["properties"]["url"], f["properties"]["count"])
             for f in json.loads(index.read_text())["features"] if shape(f["geometry"]).contains(area)]
    return sorted(found, key=lambda c: collection_year(c[0]), reverse=True)


def ept_nodes(base: str, meta, box):
    """The octree's nodes whose horizontal bounds intersect `box` (in its coordinates), with their point counts."""
    b = meta["bounds"]
    nodes, pending = [], ["0-0-0-0"]
    while pending:
        key = pending.pop()
        r = requests.get(f"{base}/ept-hierarchy/{key}.json", headers=HEADERS, timeout=60)
        r.raise_for_status()
        for node, count in r.json().items():
            d, x, y, _ = map(int, node.split("-"))
            size = (b[3] - b[0]) / 2**d
            x0, y0 = b[0] + x * size, b[1] + y * size
            if x0 > box[2] or x0 + size < box[0] or y0 > box[3] or y0 + size < box[1]:
                continue
            if count == -1:
                if node != key:
                    pending.append(node)
            elif count > 0:
                nodes.append((node, count))
    return nodes


def ept_chunks(url: str, bounds, crs: CRS, cache: pathlib.Path, workers=16):
    """The returns of an EPT collection over the area, as chunks for `accumulate`."""
    import laspy
    base = url.rsplit("/", 1)[0]
    meta = requests.get(url, headers=HEADERS, timeout=60).json()
    srs = meta["srs"]
    ept_crs = CRS.from_user_input(f"{srs['authority']}:{srs['horizontal']}" if "horizontal" in srs else srs["wkt"])
    to_site = Transformer.from_crs(ept_crs, crs, always_xy=True)
    to_ept = Transformer.from_crs(crs, ept_crs, always_xy=True)
    west, south, east, north = bounds
    xs, ys = to_ept.transform([west, east, west, east], [south, south, north, north])
    box = (min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 10)
    nodes = ept_nodes(base, meta, box)
    total = sum(c for _, c in nodes)
    print(f"lidar: {len(nodes)} octree nodes, {total:,} points over the area's bounds")
    folder = cache / base.rsplit("/", 1)[-1]
    folder.mkdir(parents=True, exist_ok=True)

    def fetch(node):
        path = folder / f"{node}.laz"
        if not path.exists():
            for attempt in range(5):
                try:
                    r = requests.get(f"{base}/ept-data/{node}.laz", headers=HEADERS, timeout=120)
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 4:
                        raise
            tmp = path.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.rename(path)
        las = laspy.read(path)
        return np.asarray(las.x), np.asarray(las.y), np.asarray(las.z), np.asarray(las.classification), \
            np.asarray(las.withheld, bool) if "withheld" in las.point_format.dimension_names else None

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for i, (x, y, z, c, withheld) in enumerate(pool.map(fetch, [n for n, _ in nodes])):
            keep = (x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3]) & ~np.isin(c, NOISE_CLASSES)
            if withheld is not None:
                keep &= ~withheld
            if keep.any():
                e, n = to_site.transform(x[keep], y[keep])
                yield e, n, z[keep], c[keep]
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(nodes)} nodes")


# --- The National Map's LAZ tiles ---------------------------------------------------------------------------------

def work_unit(item):
    """A tile's 3DEP work unit (the folder above its LAZ folder): tiles of one work unit form one collection."""
    parts = item["downloadURL"].split("/")
    return parts[parts.index("LAZ") - 1] if "LAZ" in parts else item["title"].split()[-2]


def tnm_collections(bounds, crs: CRS):
    """The work units whose tiles cover the area (by their bounding boxes), newest first: (name, tiles)."""
    lons, lats = area_lonlat(bounds, crs)
    bbox = f"{lons.min()},{lats.min()},{lons.max()},{lats.max()}"
    items = requests.get(TNM_API, params={"datasets": LPC_DATASET, "bbox": bbox, "max": 1000, "outputFormat": "JSON"},
                         headers=HEADERS, timeout=120).json()["items"]
    units = {}
    for item in items:
        units.setdefault(work_unit(item), []).append(item)
    found = []
    for unit, tiles in units.items():
        inside = np.zeros(lons.shape, bool)
        for t in tiles:
            b = t["boundingBox"]
            inside |= (lons >= b["minX"]) & (lons <= b["maxX"]) & (lats >= b["minY"]) & (lats <= b["maxY"])
        if inside.all():
            found.append((unit, [(t["downloadURL"], t.get("sizeInBytes") or 0) for t in tiles]))
    return sorted(found, key=lambda c: collection_year(c[0]), reverse=True)


def download(url: str, cache: pathlib.Path) -> pathlib.Path:
    """The file at `url`, downloaded once into `cache` (from 3DEP's S3 bucket, which mirrors most of rockyweb)."""
    path = cache / url.rsplit("/", 1)[-1]
    if path.exists():
        return path
    cache.mkdir(parents=True, exist_ok=True)
    s3 = url.replace("https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/", "https://prd-tnm.s3.amazonaws.com/StagedProducts/")
    for source in dict.fromkeys([s3, url]):
        try:
            with requests.get(source, stream=True, headers=HEADERS, timeout=120) as r:
                r.raise_for_status()
                tmp = path.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 22):
                        f.write(chunk)
                tmp.rename(path)
                return path
        except requests.RequestException:
            continue
    sys.exit(f"could not download {url} (rockyweb throttles: download it into {cache} by other means, and rerun)")


def tile_chunks(paths, bounds, crs: CRS):
    """The returns of LAZ tiles over the area, as chunks for `accumulate`."""
    import laspy
    west, south, east, north = bounds
    for path in paths:
        with laspy.open(path) as f:
            file_crs = f.header.parse_crs()
            if file_crs is None:
                sys.exit(f"{path} has no coordinate system")
            horizontal = file_crs.sub_crs_list[0] if file_crs.is_compound else file_crs
            to_site = Transformer.from_crs(horizontal, crs, always_xy=True)
            fx, fy = Transformer.from_crs(crs, horizontal, always_xy=True).transform(
                [west, east, west, east], [south, south, north, north])
            box = (min(fx) - 10, min(fy) - 10, max(fx) + 10, max(fy) + 10)
            for chunk in f.chunk_iterator(5_000_000):
                x, y, c = np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.classification)
                keep = (x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3]) & ~np.isin(c, NOISE_CLASSES)
                if "withheld" in chunk.point_format.dimension_names:
                    keep &= ~np.asarray(chunk.withheld, bool)
                if keep.any():
                    e, n = to_site.transform(x[keep], y[keep])
                    yield e, n, np.asarray(chunk.z)[keep], c[keep]
        print(f"  {path.name}")


# --- rasterization ------------------------------------------------------------------------------------------------

def accumulate(chunks, bounds, res: float):
    """The highest return of each `res` cell of the area (NaN where none), its class, and the mean height of its
    ground returns (NaN where none), from chunks of (east, north, height, class) in the area's coordinates. Heights
    are in the point cloud's unit and datum."""
    west, south, east, north = bounds
    w, h = int(round((east - west) / res)), int(round((north - south) / res))
    top = np.full(h * w, -np.inf)
    cls = np.zeros(h * w, np.uint8)
    ground_sum = np.zeros(h * w)
    ground_n = np.zeros(h * w)
    points = 0
    for e, n, z, c in chunks:
        col = np.floor((e - west) / res).astype(np.int64)
        row = np.floor((north - n) / res).astype(np.int64)
        inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        cell, z, c = row[inside] * w + col[inside], z[inside], c[inside]
        if not len(cell):
            continue
        points += len(cell)
        g = c == 2
        ground_sum += np.bincount(cell[g], weights=z[g], minlength=h * w)
        ground_n += np.bincount(cell[g], minlength=h * w)
        # each cell's highest return of the chunk, then of all chunks so far
        order = np.lexsort((z, cell))
        cell, z, c = cell[order], z[order], c[order]
        last = np.r_[cell[1:] != cell[:-1], True]
        cell, z, c = cell[last], z[last], c[last]
        higher = z > top[cell]
        top[cell[higher]] = z[higher]
        cls[cell[higher]] = c[higher]
    top[~np.isfinite(top)] = np.nan
    ground = np.where(ground_n > 0, ground_sum / np.maximum(ground_n, 1), np.nan)
    return top.reshape(h, w), cls.reshape(h, w), ground.reshape(h, w), points


def surface_model(top, cls, ground, dem, fill_distance=3, spike_height=8.0):
    """A complete surface model, in the DEM's unit and datum: `top`'s heights in metres or US survey feet (whichever
    makes its ground returns fit the DEM), offset by the median difference of its ground returns from the DEM; its
    holes filled from the nearest return within `fill_distance` cells (dark roofs, gaps between scan lines), else
    from the DEM (water, which returns little; class NO_DATA_CLASS); and its spikes cut down to their surroundings (the
    surface's grey opening): cells with at most two of their 24 neighbours (5 x 5) less than `spike_height` m below
    them (unflagged noise: birds, spurious returns hundreds of metres up; also lone poles and masts). Returns the
    surface, its classes, the height unit (m), the datum offset removed (m), the fraction filled from the DEM and
    the fraction of spikes."""
    both = np.isfinite(ground)
    if both.sum() < 100:
        sys.exit("the point cloud has (almost) no ground returns over the area")
    fits = []
    for unit in (1.0, 1200 / 3937):
        residual = ground[both] * unit - dem[both]
        offset = float(np.median(residual))
        fits.append((float(np.median(np.abs(residual - offset))), unit, offset))
    _, unit, offset = min(fits)
    dsm = top * unit - offset
    hole = ~np.isfinite(dsm)
    distance, (ri, ci) = ndimage.distance_transform_edt(hole, return_indices=True)
    near = hole & (distance <= fill_distance)
    dsm[near] = dsm[ri[near], ci[near]]
    cls = cls.copy()
    cls[near] = cls[ri[near], ci[near]]
    far = hole & ~near
    dsm[far] = dem[far]
    cls[far] = NO_DATA_CLASS
    padded = np.pad(dsm, 2, mode="edge")
    h, w = dsm.shape
    near = sum((padded[2 + dr:2 + dr + h, 2 + dc:2 + dc + w] > dsm - spike_height).astype(np.uint8)
               for dr in range(-2, 3) for dc in range(-2, 3) if dr or dc)
    spike = near <= 2
    opened = ndimage.grey_opening(dsm, size=3)
    dsm[spike] = np.minimum(dsm, opened)[spike]
    # nothing below the bare earth (multipath returns, the DEM's own smoothing)
    dsm = np.maximum(dsm, dem)
    return dsm.astype(np.float32), cls, unit, offset, float(far.mean()), float(spike.mean())


def fetch(bounds, crs: CRS, dem, cache: pathlib.Path, project=None, res=1.0):
    """The area's surface model and classes (see surface_model), and a description of their source."""
    ept = ept_collections(bounds, crs, cache)
    tnm = [] if ept and not project else tnm_collections(bounds, crs)
    choices = [("ept", name, url) for name, url, _ in ept] + [("tiles", name, tiles) for name, tiles in tnm]
    if project:
        choices = [c for c in choices if project in c[1]]
    if not choices:
        sys.exit("no 3DEP lidar collection covers the whole area" + (f" named {project}" if project else ""))
    kind, name, source = choices[0]
    print(f"lidar: 3DEP {name} ({'Entwine Point Tiles' if kind == 'ept' else 'LAZ tiles'})")
    if kind == "ept":
        chunks = ept_chunks(source, bounds, crs, cache)
        files = [source]
    else:
        print(f"lidar: {len(source)} tile(s), {sum(s for _, s in source) / 1e9:.2f} GB")
        paths = [download(url, cache / name) for url, _ in source]
        chunks = tile_chunks(paths, bounds, crs)
        files = [url for url, _ in source]
    top, cls, ground, points = accumulate(chunks, bounds, res)
    dsm, cls, unit, offset, filled, spikes = surface_model(top, cls, ground, dem)
    info = {"source": "USGS 3DEP lidar point cloud, " + ("Entwine Point Tiles (AWS)" if kind == "ept" else "LAZ tiles"),
            "collection": name, "files": files, "returns": points, "height_unit_m": unit,
            "datum_offset_m": round(offset, 3), "filled_from_dem": round(filled, 5), "spikes_removed": round(spikes, 6)}
    return dsm, cls, info
