#==================================================
CITY = "bologna"
#==================================================

import warnings; warnings.filterwarnings("ignore")
import os, sys, multiprocessing
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, box
from shapely.ops import unary_union
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage import binary_opening
import osmnx as ox

project_root = Path("/Users/terezasaskova/Library/Mobile Documents/com~apple~CloudDocs/3-30-300-index")
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from config.config import connect_db, gdf_to_duckdb, get_city_cfg, get_output_paths, CRS_WGS, CRS_METRIC

cfg   = get_city_cfg(CITY)
paths = get_output_paths(CITY)
CRS   = CRS_METRIC

# ── constants ─────────────────────────────────────────────────────────────────
NDVI_THRESH          = 0.05
MIN_AREA_M2          = 1.0
OPEN_ITERS           = 1
SIMPLIFY_M           = 0.2
OSM_NEGATIVE_BUFFER_M = 2.0

TILE_DIR    = "/Users/terezasaskova/Library/CloudStorage/GoogleDrive-tsaskova@fbk.eu/My Drive/3-30-300/data/Italy/Bologna/bologna_nir_tiles"
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

EXCLUSION_UNION = None  # set in __main__ before fork


# ── OSM exclusion ─────────────────────────────────────────────────────────────

def build_osm_exclusion(city: str) -> gpd.GeoDataFrame | None:
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


# ── per-tile worker ───────────────────────────────────────────────────────────

def process_tile(tile_path: str) -> str | None:
    p        = Path(tile_path)
    out_path = SCRATCH_DIR / (p.stem + ".gpkg")
    if out_path.exists():
        return str(out_path)

    with rasterio.open(tile_path) as src:
        tags = src.tags()
        try:
            bbox = box(float(tags["true_xmin"]), float(tags["true_ymin"]),
                       float(tags["true_xmax"]), float(tags["true_ymax"]))
        except KeyError:
            b = src.bounds
            bbox = box(b.left, b.bottom, b.right, b.top)
        data, transform, crs = src.read([1, 2], out_dtype=np.float32), src.transform, src.crs

    nir, red = data[0], data[1]
    denom    = nir + red
    ndvi     = np.where(denom > 0, (nir - red) / denom, 0.0)
    mask     = binary_opening((ndvi > NDVI_THRESH).astype(np.uint8), iterations=OPEN_ITERS)
    if not mask.any():
        return None

    geoms = [shape(g) for g, _ in shapes(mask.astype(np.uint8), mask=mask.astype(np.uint8),
                                          transform=transform, connectivity=8)]
    if not geoms:
        return None

    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs=crs)
    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_M, preserve_topology=True).buffer(0).intersection(bbox)
    gdf = gdf[~gdf.geometry.is_empty].to_crs(CRS_METRIC)
    gdf = gdf[gdf.geometry.area >= MIN_AREA_M2].copy()
    if gdf.empty:
        return None

    if EXCLUSION_UNION is not None and box(*gdf.total_bounds).intersects(EXCLUSION_UNION):
        local_excl = EXCLUSION_UNION.intersection(box(*gdf.total_bounds))
        gdf["geometry"] = gdf.geometry.difference(local_excl).buffer(0)
        gdf = gdf[~gdf.geometry.is_empty].explode(index_parts=False).reset_index(drop=True)
        gdf = gdf[gdf.geometry.area >= MIN_AREA_M2].copy()
        if gdf.empty:
            return None

    gdf["city"] = CITY
    gdf.to_crs(CRS_WGS).to_file(out_path, driver="GPKG", layer="canopy")
    return str(out_path)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    tile_paths = [p for p in sorted(Path(TILE_DIR).glob("tile_*.tif")) if p.stat().st_size > 10_000]
    print(f"City: {CITY}  |  Tiles: {len(tile_paths):,}")

    exclusion_gdf = build_osm_exclusion(CITY)
    EXCLUSION_UNION = unary_union(exclusion_gdf.geometry.values) if exclusion_gdf is not None else None
    print(f"OSM exclusion: {'ready' if EXCLUSION_UNION else 'none'}")

    already_done = len(list(SCRATCH_DIR.glob("*.gpkg")))
    if already_done:
        print(f"Resuming — {already_done:,} tiles already done.")

    gpkg_paths = []
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as exe:
        futures = {exe.submit(process_tile, str(p)): p for p in tile_paths}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Canopy"):
            try:
                r = f.result()
                if r:
                    gpkg_paths.append(r)
            except Exception as e:
                print(f"[warn] {futures[f].name}: {e}")

    print(f"Tiles with canopy: {len(gpkg_paths):,} / {len(tile_paths):,}")
    if not gpkg_paths:
        sys.exit(0)

    # merge in chunks
    CHUNK  = 500
    chunks = [gpd.GeoDataFrame(pd.concat([gpd.read_file(p) for p in gpkg_paths[i:i+CHUNK]],
              ignore_index=True)) for i in range(0, len(gpkg_paths), CHUNK)]
    final  = gpd.GeoDataFrame(pd.concat(chunks, ignore_index=True), crs=chunks[0].crs)
    final["area_m2"] = final.geometry.to_crs(CRS_METRIC).area
    final  = final[final["area_m2"] >= MIN_AREA_M2].copy()
    print(f"Total polygons: {len(final):,}")

    final.to_file(f"data/{CITY}/canopy_polygons.gpkg", driver="GPKG", layer="canopy")

    con = connect_db()
    con.execute(f"DROP TABLE IF EXISTS {DB_TABLE}")
    gdf_to_duckdb(con, final, DB_TABLE)
    con.close()
    print(f"Saved: {DB_TABLE}")