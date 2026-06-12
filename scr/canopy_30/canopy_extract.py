import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--city", default="bologna")
_args = _parser.parse_args()
CITY = _args.city

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
from shapely.validation import make_valid
import duckdb
import osmnx as ox
from tqdm import tqdm

ROOT_PATH = "/workspace/3-30-300-index/"
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
EXG_THRESH            = 0.05   # Excess Green threshold for RGB cities
MIN_AREA_M2 = 0.05
SIMPLIFY_M            = 0.5
OSM_NEGATIVE_BUFFER_M = 2.0
TILE_TIMEOUT_S        = 120

TILE_DIR           = f"/workspace/3-30-300-index/data/{CITY}_nir_tiles_local"
DB_TABLE           = f"preprocessing.canopy_{CITY}"
SCRATCH_DIR        = Path(f"/workspace/3-30-300-index/data/canopy_{CITY}_tmp")
OSM_EXCLUSION_PATH = Path(f"/workspace/3-30-300-index/data/osm_exclusion_{CITY}.gpkg")

SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

OSM_EXCLUDE_TAGS = {
    "leisure": ["pitch", "track", "golf_course", "stadium"],
    "landuse": ["agriculture", "crop","farmland", "farmyard", "meadow", "grass", "recreation_ground",
                "village_green", "orchard", "vineyard", "allotments"],
    "natural": ["grassland", "fell", "heath", "scrub"],
    "landcover": ["crops", "grass"],
}
# Shared across child processes via fork
CITY_BOUNDARY = None


# ── Vegetation indices ────────────────────────────────────────────────────────
def ndvi_mask(nir: np.ndarray, red: np.ndarray, thresh: float) -> np.ndarray:
    denom = nir + red
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)
    return (ndvi > thresh).astype(np.uint8)


def exg_mask(r: np.ndarray, g: np.ndarray, b: np.ndarray, thresh: float) -> np.ndarray:
    """Excess Green index: 2G - R - B (normalised by total)."""
    total = r + g + b
    with np.errstate(invalid="ignore", divide="ignore"):
        r_n = np.where(total > 0, r / total, 0.0)
        g_n = np.where(total > 0, g / total, 0.0)
        b_n = np.where(total > 0, b / total, 0.0)
    exg = 2 * g_n - r_n - b_n
    return (exg > thresh).astype(np.uint8)


def get_veg_mask(data: np.ndarray, nir_available: bool) -> np.ndarray:
    if nir_available:
        return ndvi_mask(data[0], data[1], NDVI_THRESH)
    else:
        return exg_mask(data[0], data[1], data[2], EXG_THRESH)


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


# ── Per-tile worker ───────────────────────────────────────────────────────────
def process_tile(tile_path: str) -> str | None:
    p = Path(tile_path)
    out_gpkg = SCRATCH_DIR / (p.stem + "_vector.gpkg")

    if out_gpkg.exists() and out_gpkg.stat().st_size > 4096:
        return str(out_gpkg)

    try:
        with rasterio.open(tile_path) as src:
            tags = src.tags()
            nir_available = tags.get("nir", "False") == "True"
            n_bands = min(src.count, 3)
            data = src.read(list(range(1, n_bands + 1)), out_dtype=np.float32)
            transform = src.transform
            crs = src.crs

            if CITY_BOUNDARY is not None:
                tile_box = box(*src.bounds)
                tile_gdf = gpd.GeoDataFrame({"geometry": [tile_box]}, crs=crs).to_crs(CRS_METRIC)
                if not tile_gdf.geometry[0].intersects(CITY_BOUNDARY):
                    return None
    except Exception as e:
        raise RuntimeError(f"rasterio open failed: {e}")

    mask = get_veg_mask(data, nir_available)
    if not mask.any():
        return None

    results = (
        {"geometry": shape(s), "properties": {"DN": int(v)}}
        for s, v in shapes(mask, mask=(mask == 1), transform=transform)
    )
    gdf = gpd.GeoDataFrame.from_features(list(results), crs=crs).to_crs(CRS_METRIC)
    if gdf.empty:
        return None

    if CITY_BOUNDARY is not None:
        gdf = gdf[gdf.geometry.intersects(CITY_BOUNDARY)].copy()

    if not gdf.empty:
        gdf.to_file(str(out_gpkg), driver="GPKG", layer="canopy")
        return str(out_gpkg)

    return None


# ── DuckDB post-process ───────────────────────────────────────────────────────
# ── DuckDB post-process ───────────────────────────────────────────────────────
def duckdb_postprocess(vector_paths, exclusion_gdf):
    process_con = duckdb.connect()
    process_con.execute("INSTALL spatial; LOAD spatial;")

    valid_paths = [
        str(Path(p).resolve())
        for p in vector_paths
        if Path(p).exists() and Path(p).stat().st_size > 4096
    ]

    print(f"DuckDB: Loading {len(valid_paths):,} shards ...")
    union_sql = " UNION ALL ".join(
        f"SELECT geom FROM ST_Read('{p}') WHERE ST_Area(geom) >= {MIN_AREA_M2}"
        for p in valid_paths
    )
    process_con.execute(f"CREATE OR REPLACE TABLE canopy_raw AS {union_sql}")
    n_raw = process_con.execute("SELECT COUNT(*) FROM canopy_raw").fetchone()[0]
    print(f"  {n_raw:,} raw polygons")

    print("DuckDB: Simplifying ...")
    process_con.execute(f"""
        CREATE OR REPLACE TABLE canopy_final AS
        SELECT ST_SimplifyPreserveTopology(geom, {SIMPLIFY_M}) AS geom
        FROM canopy_raw
        WHERE ST_Area(geom) >= {MIN_AREA_M2}
    """)
    n_final = process_con.execute("SELECT COUNT(*) FROM canopy_final").fetchone()[0]
    print(f"  {n_final:,} polygons after simplify")

    if exclusion_gdf is not None:
        print("Applying OSM exclusion via DuckDB ...")
        excl_path = str(SCRATCH_DIR / "osm_exclusion_tmp.gpkg")
        exclusion_gdf.to_file(excl_path, driver="GPKG")
        process_con.execute(f"CREATE OR REPLACE TABLE excl AS SELECT ST_Union_Agg(geom) AS geom FROM ST_Read('{excl_path}')")
        process_con.execute("""
            CREATE OR REPLACE TABLE canopy_final AS
            SELECT ST_Difference(ST_MakeValid(c.geom), ST_MakeValid(e.geom)) AS geom
            FROM canopy_final c, excl e
            WHERE NOT ST_IsEmpty(ST_Difference(ST_MakeValid(c.geom), ST_MakeValid(e.geom)))
        """)
        n_excl = process_con.execute("SELECT COUNT(*) FROM canopy_final").fetchone()[0]
        print(f"  {n_excl:,} polygons after OSM exclusion")

    result = process_con.execute("SELECT ST_AsWKB(geom) AS geom FROM canopy_final").fetchdf()
    process_con.close()
    from shapely import wkb
    result["geometry"] = result["geom"].apply(lambda b: wkb.loads(bytes(b)))
    gdf = gpd.GeoDataFrame(result.drop(columns=["geom"]), geometry="geometry", crs=CRS_METRIC)
    return gdf

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    n_workers = max(1, os.cpu_count() // 2 - 1)

    # ── Load boundary (gpkg preferred to avoid DuckDB lock contention) ────
    boundary_gpkg = Path(f"/workspace/3-30-300-index/data/{CITY}_boundary.gpkg")
    if boundary_gpkg.exists():
        boundary_gdf = gpd.read_file(str(boundary_gpkg))
    else:
        init_con = connect_db()
        boundary_gdf = load_table(init_con, "raw", f"{CITY}_boundary")
        init_con.close()

    boundary_gdf  = boundary_gdf.to_crs(CRS_METRIC)
    CITY_BOUNDARY = unary_union(boundary_gdf.geometry.values)
    print(f"City boundary loaded: {CITY_BOUNDARY.geom_type}")

    tile_paths = [p for p in sorted(Path(TILE_DIR).glob("tile_*.tif")) if p.stat().st_size > 10_000]
    print(f"City: {CITY}  |  Source tiles: {len(tile_paths):,}")

    # detect NIR from first tile
    with rasterio.open(str(tile_paths[0])) as src:
        nir_flag = src.tags().get("nir", "False") == "True"
    print(f"Vegetation index: {'NDVI (NIR)' if nir_flag else 'ExG (RGB)'}")

    exclusion_gdf = build_osm_exclusion(CITY)
    print(f"OSM exclusion: {'ready' if exclusion_gdf is not None else 'none'}")

    existing_shards = [p for p in SCRATCH_DIR.glob("*_vector.gpkg") if p.stat().st_size > 4096]

    if len(existing_shards) >= len(tile_paths) * 0.5:
        print(f"Skipping raster stage: {len(existing_shards):,} shards already in SCRATCH_DIR")
        vector_paths = [str(p) for p in existing_shards]
    else:
        print(f"Resuming from {len(existing_shards):,} existing shards …" if existing_shards else "Starting raster stage …")
        t_infer = time.time()
        vector_paths = []
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            futures = {exe.submit(process_tile, str(p)): p for p in tile_paths}
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Vectorizing {CITY}"):
                try:
                    r = f.result(timeout=TILE_TIMEOUT_S)
                    if r:
                        vector_paths.append(r)
                except FuturesTimeout:
                    tile_p = futures[f]
                    tqdm.write(f"[timeout] {tile_p.name}")
                    ghost = SCRATCH_DIR / (tile_p.stem + "_vector.gpkg")
                    if ghost.exists():
                        ghost.unlink(missing_ok=True)
                except Exception as e:
                    tqdm.write(f"[warn] {futures[f].name}: {e}")

        vector_paths = [str(p) for p in SCRATCH_DIR.glob("*_vector.gpkg") if p.stat().st_size > 4096]
        print(f"Vector shards: {len(vector_paths):,} / {len(tile_paths):,}  ({(time.time()-t_infer)/60:.1f} min)")

    if not vector_paths:
        print("No canopy data found. Exiting.")
        sys.exit(0)

    t_post = time.time()
    final = duckdb_postprocess(vector_paths, exclusion_gdf)
    print(f"Post-processing done in {(time.time()-t_post)/60:.1f} min")

    out_gpkg = Path(f"data/{CITY}/canopy_polygons.gpkg")
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    final.to_crs(CRS_WGS).to_file(str(out_gpkg), driver="GPKG", layer="canopy")
    print(f"Saved: {out_gpkg}  ({len(final):,} polygons)")

    save_con = connect_db()
    save_con.execute(f"DROP TABLE IF EXISTS {DB_TABLE}")
    gdf_to_duckdb(save_con, final.to_crs(CRS_METRIC), DB_TABLE)
    save_con.close()
    print(f"DB committed: {DB_TABLE}")