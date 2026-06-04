# ==================================================
CITY = "bologna"
# ==================================================

import warnings; warnings.filterwarnings("ignore")
import os, sys, time, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, box
from shapely.ops import unary_union
import numba
import duckdb
import osmnx as ox
from tqdm import tqdm

ROOT_PATH = "/Users/terezasaskova/Library/Mobile Documents/com~apple~CloudDocs/3-30-300-index"
sys.path.insert(0, ROOT_PATH)
os.chdir(ROOT_PATH)

from config.config import (
    connect_db, gdf_to_duckdb, load_table,
    get_city_cfg, get_output_paths,
    CRS_WGS, CRS_METRIC, DB_PATH
)

cfg   = get_city_cfg(CITY)
paths = get_output_paths(CITY)
CRS   = CRS_METRIC

NDVI_THRESH           = 0.3
MIN_AREA_M2           = 2.0
SIMPLIFY_M            = 0.5
OSM_NEGATIVE_BUFFER_M = 2.0
TILE_TIMEOUT_S        = 120

TILE_DIR  = "/Users/terezasaskova/Downloads/bologna_nir_tiles_local"
DB_TABLE = f"preprocessing.canopy_{CITY}"

# BULLETPROOF WORKSPACE: Set completely outside iCloud Drive to eliminate space/sync issues
SCRATCH_DIR = Path(f"/Users/terezasaskova/Downloads/canopy_{CITY}_tmp")
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
OSM_EXCLUSION_PATH = Path(f"data/{CITY}/osm_exclusion.gpkg")

OSM_EXCLUDE_TAGS = {
    "leisure": ["pitch", "track", "golf_course", "stadium"],
    "landuse": ["farmland", "farmyard", "meadow", "grass", "recreation_ground",
                "village_green", "orchard", "vineyard", "allotments"],
    "natural": ["grassland", "fell", "heath"],
}

# Shared across child processes via fork memory space
CITY_BOUNDARY = None

# Numba NDVI 
@numba.njit(parallel=True, cache=True)
def ndvi_mask(nir: np.ndarray, red: np.ndarray, thresh: float) -> np.ndarray:
    """Returns uint8 binary mask: 1 where NDVI > thresh, else 0."""
    out = np.zeros(nir.shape, dtype=np.uint8)
    for i in numba.prange(nir.shape[0]):
        for j in range(nir.shape[1]):
            d = nir[i, j] + red[i, j]
            if d > 0 and (nir[i, j] - red[i, j]) / d > thresh:
                out[i, j] = 1
    return out


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

# ── Per-tile: compute NDVI mask AND vectorise locally ──────────────────────────
def process_tile(tile_path: str) -> str | None:
    p = Path(tile_path)
    out_gpkg = SCRATCH_DIR / (p.stem + "_vector.gpkg")

    # Skip if vector shard is already processed and sound
    if out_gpkg.exists() and out_gpkg.stat().st_size > 4096:
        return str(out_gpkg)

    try:
        with rasterio.open(tile_path) as src:
            data = src.read([1, 2], out_dtype=np.float32)
            transform = src.transform
            crs = src.crs
            
            # Fast bounding box intersection check against city boundary
            if CITY_BOUNDARY is not None:
                tile_box = box(*src.bounds)
                tile_gdf = gpd.GeoDataFrame({"geometry": [tile_box]}, crs=crs).to_crs(CRS_METRIC)
                if not tile_gdf.geometry[0].intersects(CITY_BOUNDARY):
                    return None
    except Exception as e:
        raise RuntimeError(f"rasterio open failed: {e}")

    # Calculate mask via Numba
    mask = ndvi_mask(data[0], data[1], NDVI_THRESH)
    if not mask.any():
        return None

    # Vectorise the local matrix boundaries natively using C endpoints
    results = (
        {"geometry": shape(s), "properties": {"DN": int(v)}}
        for s, v in shapes(mask, mask=(mask == 1), transform=transform)
    )
    
    # Pack up features directly to Metric projection
    gdf = gpd.GeoDataFrame.from_features(list(results), crs=crs).to_crs(CRS_METRIC)
    if gdf.empty:
        return None
        
    if CITY_BOUNDARY is not None:
        gdf = gdf[gdf.geometry.intersects(CITY_BOUNDARY)].copy()

    if not gdf.empty:
        gdf.to_file(str(out_gpkg), driver="GPKG", layer="canopy")
        return str(out_gpkg)
        
    return None


def duckdb_postprocess(vector_paths: list[str], exclusion_gdf: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    process_con = duckdb.connect()
    process_con.execute("INSTALL spatial; LOAD spatial;")

    valid_paths = []
    for path_str in vector_paths:
        p = Path(path_str)
        if p.exists() and p.is_file() and p.stat().st_size > 4096:
            valid_paths.append(str(p.resolve()))

    if not valid_paths:
        raise RuntimeError("No valid, uncorrupted vector shards found to post-process.")

    print(f"DuckDB: Streaming and parsing {len(valid_paths):,} valid vector shards...")

    union_sql = " UNION ALL ".join(
        f"SELECT geom FROM ST_Read('{p}') WHERE ST_Area(geom) >= {MIN_AREA_M2}"
        for p in valid_paths
    )
    process_con.execute(f"CREATE OR REPLACE TABLE canopy_raw AS {union_sql}")

    print("DuckDB: Dissolving shard boundaries...")
    process_con.execute("""
        CREATE OR REPLACE TABLE canopy_dissolved AS
        SELECT ST_Union_Agg(geom) AS geom
        FROM canopy_raw
    """)

    print("DuckDB: Exploding multipart to singlepart...")
    process_con.execute(f"""
        CREATE OR REPLACE TABLE canopy_final AS
        SELECT ST_SimplifyPreserveTopology(geom, {SIMPLIFY_M}) AS geom
        FROM (
            SELECT UNNEST(ST_Dump(geom)).geom AS geom
            FROM canopy_dissolved
        )
        WHERE ST_Area(geom) >= {MIN_AREA_M2}
    """)

    result = process_con.execute("SELECT ST_AsWKB(geom) AS geom FROM canopy_final").fetchdf()
    process_con.close()

    from shapely import wkb
    result["geometry"] = result["geom"].apply(lambda b: wkb.loads(bytes(b)))
    gdf = gpd.GeoDataFrame(result.drop(columns=["geom"]), geometry="geometry", crs=CRS_METRIC)

    if exclusion_gdf is not None:
        print("Applying OSM exclusion mask...")
        excl_union = unary_union(exclusion_gdf.geometry.values)
        gdf["geometry"] = gdf.geometry.difference(excl_union)
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

    return gdf

# ── Main Process Controller ───────────────────────────────────────────────────
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    n_workers = max(1, os.cpu_count() - 1)

    # Isolated scope initialization block
    init_con = connect_db()
    boundary_gdf  = load_table(init_con, "raw", f"{CITY}_boundary")
    init_con.close()
    
    boundary_gdf  = boundary_gdf.to_crs(CRS_METRIC)
    CITY_BOUNDARY = unary_union(boundary_gdf.geometry.values)
    print(f"City boundary loaded: {CITY_BOUNDARY.geom_type}")

    tile_paths = [p for p in sorted(Path(TILE_DIR).glob("tile_*.tif")) if p.stat().st_size > 10_000]
    print(f"City: {CITY}  |  Total source tiles found: {len(tile_paths):,}")

    exclusion_gdf = build_osm_exclusion(CITY)
    print(f"OSM exclusion reference matrix: {'ready' if exclusion_gdf is not None else 'none'}")

    already_done = len(list(SCRATCH_DIR.glob("*_vector.gpkg")))
    if already_done:
        print(f"Resuming pipeline -- {already_done:,} vector shards found in workspace.")

    # ── Stage 1: Parallel extraction and vectorization ───────────────────────
    t_infer = time.time()
    vector_paths = []
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(process_tile, str(p)): p for p in tile_paths}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing & Vectorizing Tiles"):
            try:
                r = f.result(timeout=TILE_TIMEOUT_S)
                if r:
                    vector_paths.append(r)
            except FuturesTimeout:
                tile_p = futures[f]
                print(f"[timeout] {tile_p.name} -- abandoned execution thread")
                # Immediately destroy half-written ghost file to keep directory untainted
                ghost_file = SCRATCH_DIR / (tile_p.stem + "_vector.gpkg")
                if ghost_file.exists():
                    try:
                        ghost_file.unlink()
                    except Exception:
                        pass
            except Exception as e:
                print(f"[warn] {futures[f].name}: {e}")

    # Gather clean list of completed shards from disk
    vector_paths = [str(p) for p in SCRATCH_DIR.glob("*_vector.gpkg") if p.stat().st_size > 4096]
    print(f"Vector shards generated: {len(vector_paths):,} / {len(tile_paths):,} ({(time.time()-t_infer)/60:.1f} min)")

    if not vector_paths:
        print("No active canopy data found in source matrix tiles. Shutting down pipeline.")
        sys.exit(0)

    # ── Stage 2: DuckDB Vector Assembly and Post-processing ──────────────────
    t_post = time.time()
    final = duckdb_postprocess(vector_paths, exclusion_gdf)

    # ── Stage 3: Save Consolidated Outputs ────────────────────────────────────
    out_gpkg = Path(f"data/{CITY}/canopy_polygons.gpkg")
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    final.to_crs(CRS_WGS).to_file(str(out_gpkg), driver="GPKG", layer="canopy")
    print(f"Saved definitive tracking asset to: {out_gpkg}")

    save_con = connect_db()
    save_con.execute(f"DROP TABLE IF EXISTS {DB_TABLE}")
    gdf_to_duckdb(save_con, final, DB_TABLE)
    save_con.close()
    print(f"Database deployment transaction committed to: {DB_TABLE}")