# ==================================================
CITY = "bologna"
# ==================================================

import warnings; warnings.filterwarnings("ignore")
import os, sys, time, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeout
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, box
from shapely.ops import unary_union
from shapely.validation import make_valid
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import binary_opening, generate_binary_structure
import osmnx as ox

ROOT_PATH = "/Users/terezasaskova/Library/Mobile Documents/com~apple~CloudDocs/3-30-300-index"
sys.path.insert(0, ROOT_PATH)
os.chdir(ROOT_PATH)

from config.config import (
    connect_db, gdf_to_duckdb, load_table,
    get_city_cfg, get_output_paths,
    CRS_WGS, CRS_METRIC,
)

cfg   = get_city_cfg(CITY)
paths = get_output_paths(CITY)
CRS   = CRS_METRIC

NDVI_THRESH = 0.3    # standard canopy threshold
MIN_AREA_M2 = 2.0    # raised from 1m² — removes more noise without losing real canopy
OPEN_ITERS  = 1      # morphological opening iterations
SIMPLIFY_M  = 0.5    # raised from 0.2m — reduces vertex count, speeds up dissolve
OSM_NEGATIVE_BUFFER_M = 2.0    # shrink OSM exclusion zones by 2m to avoid overfiltering
TILE_TIMEOUT_S = 120    # raised from 60s — dense tiles were timing out
CHUNK_SIZE = 200    # chunk size for final unary_union to avoid memory spike

TILE_DIR    = "/Users/terezasaskova/Downloads/bologna_nir_tiles_local"
DB_TABLE    = f"preprocessing.canopy_{CITY}"
SCRATCH_DIR = Path(f"data/{CITY}/canopy_tiles_tmp")
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
OSM_EXCLUSION_PATH = Path(f"data/{CITY}/osm_exclusion.gpkg")

OSM_EXCLUDE_TAGS = {
    "leisure": ["pitch", "track", "golf_course", "stadium"],
    "landuse": ["farmland", "farmyard", "meadow", "grass", "recreation_ground",
                "village_green", "orchard", "vineyard", "allotments"],
    "natural": ["grassland", "fell", "heath"],
}

# 3x3 cross-shaped structuring element for morphological opening —
# removes isolated noise pixels without eroding real canopy patches
MORPH_STRUCT = generate_binary_structure(2, 1)

# Globals set in __main__ before fork
CITY_BOUNDARY = None

def build_osm_exclusion(city: str) -> gpd.GeoDataFrame | None:
    """Fetch OSM non-tree landuse polygons and cache to disk.
    Used post-merge to remove NDVI false positives (grass, farmland, etc)."""
    if OSM_EXCLUSION_PATH.exists():
        return gpd.read_file(OSM_EXCLUSION_PATH).to_crs(CRS_METRIC)

    gdfs = []
    for key, values in OSM_EXCLUDE_TAGS.items():
        for value in values:
            try:
                gdf = ox.features_from_place(f"{city}, Italy", tags={key: value})
                polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])][["geometry"]].copy()
                if not polys.empty:
                    polys["osm_tag"] = f"{key}={value}"
                    gdfs.append(polys)
            except Exception:
                pass

    if not gdfs:
        return None

    exclusion = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=CRS_WGS).to_crs(CRS_METRIC)
    exclusion["geometry"] = exclusion.geometry.buffer(-OSM_NEGATIVE_BUFFER_M)
    exclusion = exclusion[~exclusion.geometry.is_empty].copy()
    OSM_EXCLUSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    exclusion.to_file(OSM_EXCLUSION_PATH, driver="GPKG")
    return exclusion


def apply_osm_exclusion(final: gpd.GeoDataFrame, exclusion_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Subtract OSM non-tree areas from canopy polygons.
    Applied once post-merge — no per-tile cost."""
    excl_union = unary_union(exclusion_gdf.geometry.values)
    final["geometry"] = final.geometry.difference(excl_union).buffer(0)
    final = final[~final.geometry.is_empty].explode(index_parts=False).reset_index(drop=True)
    return final[final.geometry.area >= MIN_AREA_M2].copy()

def process_tile(tile_path: str) -> str | None:
    """Extract canopy polygons from a single tile using NDVI thresholding.

    Pipeline:
      1. Read NIR (band 1) + Red (band 2)
      2. Compute NDVI, threshold, morphological opening
      3. Vectorize → simplify → reproject → area filter
      4. Clip to city boundary
      5. Save to scratch GeoPackage
    """
    t0 = time.time()
    p  = Path(tile_path)
    out_path = SCRATCH_DIR / (p.stem + ".gpkg")

    if out_path.exists() and out_path.stat().st_size > 1_000:
        return str(out_path)

    try:
        with rasterio.open(tile_path) as src:
            data = src.read([1, 2], out_dtype=np.float32)
            transform = src.transform
            crs       = src.crs
    except Exception as e:
        raise RuntimeError(f"rasterio open failed: {e}")
    
    nir, red = data[0], data[1]
    denom = nir + red
    ndvi  = np.where(denom > 0, (nir - red) / denom, 0.0)

    mask = binary_opening(
        (ndvi > NDVI_THRESH).astype(np.uint8),
        structure=MORPH_STRUCT,
        iterations=OPEN_ITERS,
    )
    if not mask.any():
        return None

    geoms = [
        shape(g)
        for g, _ in shapes(
            mask.astype(np.uint8),
            mask=mask.astype(np.uint8),
            transform=transform,
            connectivity=8,
        )
    ]
    if not geoms:
        return None

    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs=crs)
    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_M, preserve_topology=True).buffer(0)
    gdf = gdf[~gdf.geometry.is_empty].to_crs(CRS_METRIC)
    gdf = gdf[gdf.geometry.area >= MIN_AREA_M2].copy()
    if gdf.empty:
        return None

    if CITY_BOUNDARY is not None:
        tile_bbox = box(*gdf.total_bounds)
        if not tile_bbox.intersects(CITY_BOUNDARY):
            return None
        local_boundary = CITY_BOUNDARY.intersection(tile_bbox)
        gdf["geometry"] = gdf.geometry.intersection(local_boundary).buffer(0)
        gdf = gdf[~gdf.geometry.is_empty].explode(index_parts=False).reset_index(drop=True)
        gdf = gdf[gdf.geometry.area >= MIN_AREA_M2].copy()
        if gdf.empty:
            return None

    elapsed = time.time() - t0
    if elapsed > 15:
        print(f"[slow] {p.name}: {elapsed:.1f}s  polygons={len(gdf)}")

    gdf["city"] = CITY
    gdf.to_crs(CRS_WGS).to_file(out_path, driver="GPKG", layer="canopy")
    return str(out_path)


def _load_geoms(path: str) -> list:
    try:
        return list(gpd.read_file(path).geometry)
    except Exception:
        return []


def _chunked_union(geoms: list, chunk_size: int):
    """Two-level chunked union to avoid memory spike on large polygon sets."""
    chunks       = [geoms[i:i + chunk_size] for i in range(0, len(geoms), chunk_size)]
    intermediate = [unary_union(c) for c in tqdm(chunks, desc="Chunked union")]
    return unary_union(intermediate)


def merge_tiles(gpkg_paths: list[str], n_workers: int) -> gpd.GeoDataFrame:
    """Load per-tile GeoPackages in parallel, fix geometries, dissolve tile boundaries."""
    all_geoms = []
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(_load_geoms, p): p for p in gpkg_paths}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Loading"):
            all_geoms.extend(f.result())

    print(f"Total raw polygons before dissolve: {len(all_geoms):,}")

    # buffer(0) + make_valid before union prevents TopologyException
    all_geoms = [make_valid(g.buffer(0)) for g in all_geoms if g is not None and not g.is_empty]

    merged = _chunked_union(all_geoms, CHUNK_SIZE)
    geoms  = (
        list(merged.geoms)
        if merged.geom_type in ("MultiPolygon", "GeometryCollection")
        else [merged]
    )
    sample_crs = gpd.read_file(gpkg_paths[0]).crs
    return gpd.GeoDataFrame(geometry=geoms, crs=sample_crs)


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    n_workers = max(1, os.cpu_count() - 1)

    con = connect_db()
    boundary_gdf  = load_table(con, "raw", f"{CITY}_boundary")
    con.close()
    boundary_gdf  = boundary_gdf.to_crs(CRS_METRIC)
    CITY_BOUNDARY = unary_union(boundary_gdf.geometry.values)
    print(f"City boundary loaded: {CITY_BOUNDARY.geom_type}")

    tile_paths = [p for p in sorted(Path(TILE_DIR).glob("tile_*.tif")) if p.stat().st_size > 10_000]
    print(f"City: {CITY}  |  Tiles: {len(tile_paths):,}")

    exclusion_gdf = build_osm_exclusion(CITY)
    print(f"OSM exclusion: {'ready' if exclusion_gdf is not None else 'none'}")

    already_done = len(list(SCRATCH_DIR.glob("*.gpkg")))
    if already_done:
        print(f"Resuming — {already_done:,} tiles already done.")

    t_infer    = time.time()
    gpkg_paths = []
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(process_tile, str(p)): p for p in tile_paths}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Canopy"):
            try:
                r = f.result(timeout=TILE_TIMEOUT_S)
                if r:
                    gpkg_paths.append(r)
            except FuturesTimeout:
                print(f"[timeout] {futures[f].name} — skipped")
            except Exception as e:
                print(f"[warn] {futures[f].name}: {e}")

    print(f"Tiles with canopy: {len(gpkg_paths):,} / {len(tile_paths):,}  ({(time.time()-t_infer)/60:.1f} min)")
    if not gpkg_paths:
        sys.exit(0)

    t_merge = time.time()
    print("Merging tiles...")
    final = merge_tiles(gpkg_paths, n_workers)
    final["area_m2"] = final.geometry.area
    final = final[final["area_m2"] >= MIN_AREA_M2].copy()
    final["city"] = CITY
    print(f"Polygons after merge: {len(final):,}  ({(time.time()-t_merge)/60:.1f} min)")

    if exclusion_gdf is not None:
        t_excl = time.time()
        print("Applying OSM exclusion...")
        final = apply_osm_exclusion(final, exclusion_gdf)
        final["area_m2"] = final.geometry.area
        print(f"Polygons after OSM exclusion: {len(final):,}  ({(time.time()-t_excl)/60:.1f} min)")

    out_gpkg = Path(f"data/{CITY}/canopy_polygons.gpkg")
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    final.to_file(out_gpkg, driver="GPKG", layer="canopy")
    print(f"Saved: {out_gpkg}")

    con = connect_db()
    con.execute(f"DROP TABLE IF EXISTS {DB_TABLE}")
    gdf_to_duckdb(con, final, DB_TABLE)
    con.close()
    print(f"Saved to DB: {DB_TABLE}")