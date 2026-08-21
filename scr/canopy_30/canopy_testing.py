"""
canopy_extract_rgb.py — Extract tree canopy from RGB tiles.

Optimized with GeoParquet file streaming, mask-filtered HSV conversions,
and bounding-box-accelerated DuckDB spatial joins.

Usage:
    python canopy_extract_rgb.py --city venezia
    TEST_MODE=1 python canopy_extract_rgb.py --city genova
"""
import argparse
import os
import sys
import time
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from skimage.color import rgb2hsv

# Force PROJ environment paths
os.environ["PROJ_DATA"] = "/opt/anaconda3/envs/gis/share/proj"
os.environ["PROJ_LIB"] = "/opt/anaconda3/envs/gis/share/proj"

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, box
from shapely.ops import unary_union
import duckdb
import osmnx as ox
from tqdm import tqdm

ROOT_PATH = "/Users/terezasaskova/Library/Mobile Documents/com~apple~CloudDocs/3-30-300-index/"
sys.path.insert(0, ROOT_PATH)
os.chdir(ROOT_PATH)

from config.config import (
    connect_db, gdf_to_duckdb, load_table,
    CRS_WGS, CRS_METRIC, DB_PATH,
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

BASE_GSD_M            = 0.20
SIMPLIFY_M            = 0.5
OSM_BUFFER_M          = 2.0
WATER_BUFFER_M        = 6.0   
TILE_TIMEOUT_S        = 120
DATA_ROOT             = Path("/Users/terezasaskova/Library/Mobile Documents/com~apple~CloudDocs/3-30-300-index/data")

OSM_EXCLUDE_TAGS = {
    "leisure":  ["pitch", "track", "golf_course", "stadium"],
    "landuse":  ["agriculture", "crop", "farmland", "farmyard", "meadow", "grass",
                 "recreation_ground", "village_green", "orchard", "vineyard", "allotments"],
    "natural":  ["grassland", "fell", "heath", "scrub", "water", "wetland", "beach",
                 "bay", "strait", "coastline"],
    "landcover": ["crops", "grass"],
    "water":    ["river", "canal", "lake", "reservoir", "pond"],
    "place":    ["sea"],
    "building": True,
}

WATER_TAG_KEYS = {
    "natural": {"water", "wetland", "beach", "bay", "strait", "coastline"},
    "water":   {"river", "canal", "lake", "reservoir", "pond"},
    "place":   {"sea"},
}

OSM_INCLUDE_TAGS = {
    "landuse": ["forest", "wood"],
}

CITY_EXG_THRESH = {
    "genova":  0.13,
    "firenze": 0.09,
    "bolzano": 0.18,
    "venezia": 0.20,
    "roma":    0.18,
    "verona":  0.18,
    "milano":  0.15,
    "trento":  0.16,
    "perugia": 0.15,
    "bari":    0.10,
}

CITY_MIN_AREA_M2 = {
    "genova":  1.0,
    "venezia": 2.0,
    "verona":  2.0,
    "roma":    1.5,
    "milano":  1.5,
    "trento":  1.0,
    "perugia": 1.0,
    "bari":    1.0,
    "bolzano": 1.5,
    "firenze": 1.5,
}

CITY_HUE_RANGE = {
    "genova": (0.15, 0.42),
    "trento": (0.16, 0.42),
}
CITY_SAT_MIN = {
    "genova": 0.15,
    "trento": 0.14,
}
CITY_BRIGHTNESS_GATE = {
    "genova": 45.0,
    "trento": 44.0,
}

TILE_DIR_PATTERNS = [
    "{city}/{city}_ortho_tiles",
]

# ─────────────────────────────────────────────────────────────
# City Config Dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class CityConfig:
    city:         str
    tile_dir:     Path
    scratch_dir:  Path
    db_table:     str
    exg_thresh:   float
    base_area_m2: float
    boundary:         object = field(default=None, repr=False)
    hue_range:        Optional[tuple] = None
    sat_min:          float = 0.0
    brightness_gate:  float = 35.0

    @classmethod
    def build(cls, city: str, test_mode: bool = False) -> "CityConfig":
        base_city = city.replace("_rgb", "")

        tile_dir = _find_tile_dir(city, base_city)
        scratch_dir = DATA_ROOT / f"canopy_{city}_{'test_' if test_mode else ''}tmp"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        boundary = _load_boundary(base_city)

        return cls(
            city=city,
            tile_dir=tile_dir,
            scratch_dir=scratch_dir,
            db_table=f"preprocessing.canopy_{city}",
            exg_thresh=CITY_EXG_THRESH.get(base_city, 0.15),
            base_area_m2=CITY_MIN_AREA_M2.get(base_city, 0.5),
            boundary=boundary,
            hue_range=CITY_HUE_RANGE.get(base_city),
            sat_min=CITY_SAT_MIN.get(base_city, 0.0),
            brightness_gate=CITY_BRIGHTNESS_GATE.get(base_city, 35.0),
        )


def _find_tile_dir(city: str, base_city: str) -> Path:
    for pat in TILE_DIR_PATTERNS:
        for candidate in [
            DATA_ROOT / pat.format(city=city),
            DATA_ROOT / base_city / pat.format(city=city),
            DATA_ROOT / base_city / pat.format(city=base_city),
        ]:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No tile directory found for city='{city}'")


def _load_boundary(lookup_city: str, boundary_path: Optional[str] = None):
    if boundary_path and Path(boundary_path).exists():
        gdf = gpd.read_file(boundary_path)
    else:
        gpkg = DATA_ROOT / f"{lookup_city}_boundary.gpkg"
        if gpkg.exists():
            gdf = gpd.read_file(str(gpkg))
        else:
            con = connect_db(read_only=True)
            gdf = load_table(con, "raw", f"{lookup_city}_boundary")
            con.close()
    return unary_union(gdf.to_crs(CRS_METRIC).geometry.values)


# ─────────────────────────────────────────────────────────────
# OSM Helpers
# ─────────────────────────────────────────────────────────────

def _osm_cache_path(city: str, key: str, value) -> Path:
    safe_val = str(value).replace("/", "_")
    return DATA_ROOT / "osm_cache" / f"{city}__{key}__{safe_val}.gpkg"


def _fetch_osm_tag(city: str, key: str, value) -> Optional[gpd.GeoDataFrame]:
    cache = _osm_cache_path(city, key, value)
    if cache.exists():
        return gpd.read_file(str(cache))
    try:
        gdf = ox.features_from_place(f"{city}, Italy", tags={key: value})
        polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])][["geometry"]].copy()
        if polys.empty:
            return None
        polys["osm_tag"] = f"{key}={value}"
        cache.parent.mkdir(parents=True, exist_ok=True)
        polys.to_file(str(cache), driver="GPKG")
        return polys
    except Exception:
        return None


def _is_water_tag(key: str, value) -> bool:
    return value in WATER_TAG_KEYS.get(key, set())


def build_osm_exclusion(city: str) -> Optional[gpd.GeoDataFrame]:
    water_gdfs = []
    other_gdfs = []
    for key, values in OSM_EXCLUDE_TAGS.items():
        tag_list = values if isinstance(values, list) else [values]
        for value in tag_list:
            gdf = _fetch_osm_tag(city, key, value)
            if gdf is None:
                continue
            if _is_water_tag(key, value):
                water_gdfs.append(gdf)
            else:
                other_gdfs.append(gdf)

    if not water_gdfs and not other_gdfs:
        return None

    parts = []
    if water_gdfs:
        water = gpd.GeoDataFrame(
            pd.concat(water_gdfs, ignore_index=True), crs=CRS_WGS
        ).to_crs(CRS_METRIC)
        water["geometry"] = water.geometry.buffer(-WATER_BUFFER_M)
        parts.append(water[~water.geometry.is_empty].copy())
    if other_gdfs:
        other = gpd.GeoDataFrame(
            pd.concat(other_gdfs, ignore_index=True), crs=CRS_WGS
        ).to_crs(CRS_METRIC)
        other["geometry"] = other.geometry.buffer(-OSM_BUFFER_M)
        parts.append(other[~other.geometry.is_empty].copy())

    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=CRS_METRIC)


def build_osm_inclusion(city: str) -> Optional[gpd.GeoDataFrame]:
    gdfs = []
    for key, values in OSM_INCLUDE_TAGS.items():
        for value in values:
            gdf = _fetch_osm_tag(city, key, value)
            if gdf is not None:
                gdfs.append(gdf)
    if not gdfs:
        return None
    return gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True), crs=CRS_WGS
    ).to_crs(CRS_METRIC)


# ─────────────────────────────────────────────────────────────
# Optimized Vegetation Masking (Selective HSV)
# ─────────────────────────────────────────────────────────────

def exg_mask(r: np.ndarray, g: np.ndarray, b: np.ndarray, thresh: float,
             brightness_gate: float = 35.0,
             hue_range: Optional[tuple] = None,
             sat_min: float = 0.0) -> np.ndarray:
    total = r + g + b

    # Fast brightness filtering
    brightness_mask = total > brightness_gate
    if not brightness_mask.any():
        return np.zeros(r.shape, dtype=np.uint8)

    with np.errstate(invalid="ignore", divide="ignore"):
        r_n = np.where(total > 0, r / total, 0.0)
        g_n = np.where(total > 0, g / total, 0.0)
        b_n = np.where(total > 0, b / total, 0.0)

    exg = 2 * g_n - r_n - b_n
    mask = (exg > thresh) & brightness_mask

    # Speed Optimization: Compute HSV only on candidate pixels
    if (hue_range is not None or sat_min > 0) and mask.any():
        candidate_indices = np.where(mask)
        candidate_r = r[candidate_indices] / 255.0
        candidate_g = g[candidate_indices] / 255.0
        candidate_b = b[candidate_indices] / 255.0

        cand_rgb = np.stack([candidate_r, candidate_g, candidate_b], axis=-1)
        cand_hsv = rgb2hsv(cand_rgb)

        hue, sat = cand_hsv[..., 0], cand_hsv[..., 1]

        valid_candidates = np.ones(len(candidate_r), dtype=bool)
        if sat_min > 0:
            valid_candidates &= (sat >= sat_min)
        if hue_range is not None:
            lo, hi = hue_range
            valid_candidates &= (hue >= lo) & (hue <= hi)

        final_mask = np.zeros(r.shape, dtype=bool)
        final_mask[candidate_indices[0][valid_candidates], candidate_indices[1][valid_candidates]] = True
        return final_mask.astype(np.uint8)

    return mask.astype(np.uint8)


def veg_mask(data: np.ndarray, gsd_m: float, exg_thresh: float, base_area_m2: float,
             hue_range: Optional[tuple] = None, sat_min: float = 0.0,
             brightness_gate: float = 35.0) -> tuple[np.ndarray, float]:
    min_area = base_area_m2 * (gsd_m / BASE_GSD_M) ** 2
    mask = exg_mask(data[0], data[1], data[2], exg_thresh,
                     brightness_gate=brightness_gate, hue_range=hue_range, sat_min=sat_min)
    return mask, min_area


# ─────────────────────────────────────────────────────────────
# Per-Tile Worker (Outputs Parquet Shards)
# ─────────────────────────────────────────────────────────────

def process_tile(args: tuple) -> Optional[str]:
    (tile_path, scratch_dir, boundary, exg_thresh, base_area_m2,
     hue_range, sat_min, brightness_gate) = args
    p = Path(tile_path)

    # Speed Optimization: Write directly to Parquet
    out_parquet = Path(scratch_dir) / (p.stem + "_vector.parquet")

    if out_parquet.exists() and out_parquet.stat().st_size > 1024:
        return str(out_parquet)

    try:
        with rasterio.open(tile_path) as src:
            n_bands = min(src.count, 3)
            data    = src.read(list(range(1, n_bands + 1)), out_dtype=np.float32)
            transform = src.transform
            crs     = src.crs
            gsd_m   = abs(transform.a)

            if boundary is not None:
                tile_box = box(*src.bounds)
                tile_gdf = gpd.GeoDataFrame({"geometry": [tile_box]}, crs=crs).to_crs(CRS_METRIC)
                if not tile_gdf.geometry[0].intersects(boundary):
                    return None
    except Exception as e:
        raise RuntimeError(f"rasterio open failed: {e}")

    mask, min_area = veg_mask(data, gsd_m, exg_thresh, base_area_m2,
                               hue_range=hue_range, sat_min=sat_min, brightness_gate=brightness_gate)
    if not mask.any():
        return None

    feats = [
        {"geometry": shape(s), "properties": {"DN": int(v)}}
        for s, v in shapes(mask, mask=(mask == 1), transform=transform)
    ]
    gdf = gpd.GeoDataFrame.from_features(feats, crs=crs).to_crs(CRS_METRIC)
    if gdf.empty:
        return None

    gdf = gdf[gdf.geometry.area >= min_area].copy()
    if boundary is not None:
        gdf = gdf[gdf.geometry.intersects(boundary)].copy()

    if not gdf.empty:
        gdf.to_parquet(str(out_parquet))
        return str(out_parquet)
    return None


# ─────────────────────────────────────────────────────────────
# DuckDB Post-Processing (Native Parquet + Spatial BBoxes)
# ─────────────────────────────────────────────────────────────

def duckdb_postprocess(vector_paths, exclusion_gdf, cfg: CityConfig):
    """
    NOTE (bugfix): geopandas' `gdf.to_parquet()` writes GeoParquet, which
    embeds geo metadata describing the geometry column's encoding. DuckDB's
    spatial extension recognizes that metadata on `read_parquet()` and
    hands back the geometry column already typed as native `GEOMETRY` —
    not a raw WKB `BLOB`. Calling `ST_GeomFromWKB()` on it then fails:

        Binder Error: No function matches ... 'st_geomfromwkb(GEOMETRY(...))'
        Candidate functions: st_geomfromwkb(BLOB) -> GEOMETRY

    Fix: cast with `::GEOMETRY` instead of `ST_GeomFromWKB()`. The cast is
    a no-op when the column is already GEOMETRY and still parses correctly
    if a source ever hands back BLOB, so it's safe either way.
    """
    min_area = cfg.base_area_m2

    db_file = cfg.scratch_dir / "duckdb_temp.db"
    if db_file.exists():
        db_file.unlink()

    process_con = duckdb.connect(str(db_file))
    process_con.execute("INSTALL spatial; LOAD spatial;")

    process_con.execute(f"""
        SET threads=8;
        SET preserve_insertion_order=false;
        SET memory_limit='6GB';
        SET temp_directory='{cfg.scratch_dir}';
    """)

    print(f"DuckDB: Natively scanning {len(vector_paths):,} Parquet shards ...")

    # Speed Optimization: Wildcard Parquet scan directly via DuckDB Engine
    search_pattern = str(cfg.scratch_dir / "*_vector.parquet")
    process_con.execute(f"""
        CREATE OR REPLACE TABLE canopy_raw AS 
        SELECT geometry::GEOMETRY AS geom 
        FROM read_parquet('{search_pattern}') 
        WHERE ST_Area(geometry::GEOMETRY) >= {min_area}
    """)

    process_con.execute(f"""
        CREATE OR REPLACE TABLE canopy_final AS
        SELECT ST_SimplifyPreserveTopology(geom, {SIMPLIFY_M})::GEOMETRY AS geom
        FROM canopy_raw
        WHERE ST_Area(geom) >= {min_area}
    """)
    n_final = process_con.execute("SELECT COUNT(*) FROM canopy_final").fetchone()[0]
    print(f"  {n_final:,} polygons after simplify")

    if exclusion_gdf is not None:
        print("Applying OSM exclusion via DuckDB (filter-then-diff) ...")
        excl_path = str(cfg.scratch_dir / "osm_exclusion_tmp.parquet")
        exclusion_gdf.to_parquet(excl_path)

        process_con.execute(f"""
            CREATE OR REPLACE TABLE excl AS
            SELECT geometry::GEOMETRY AS geom FROM read_parquet('{excl_path}')
        """)

        # Bugfix (see docstring above): defensive re-cast, cheap no-op if
        # canopy_final's geom column is already GEOMETRY.
        process_con.execute("""
            CREATE OR REPLACE TABLE canopy_final AS
            SELECT geom::GEOMETRY AS geom FROM canopy_final
        """)

        process_con.execute("CREATE INDEX excl_rtree ON excl USING RTREE(geom)")
        process_con.execute("CREATE INDEX canopy_rtree ON canopy_final USING RTREE(geom)")

        # Speed Optimization: Use Bounding-Box Operator (&&)
        process_con.execute("""
            CREATE OR REPLACE TABLE canopy_clear AS
            SELECT c.geom
            FROM canopy_final c
            WHERE NOT EXISTS (
                SELECT 1 FROM excl e WHERE c.geom && e.geom AND ST_Intersects(c.geom, e.geom)
            )
        """)
        n_clear = process_con.execute("SELECT COUNT(*) FROM canopy_clear").fetchone()[0]
        print(f"  {n_clear:,} polygons untouched by exclusion (skipped diff)")

        process_con.execute("""
            CREATE OR REPLACE TABLE canopy_hit AS
            SELECT c.geom
            FROM canopy_final c
            WHERE EXISTS (
                SELECT 1 FROM excl e WHERE c.geom && e.geom AND ST_Intersects(c.geom, e.geom)
            )
        """)
        n_hit = process_con.execute("SELECT COUNT(*) FROM canopy_hit").fetchone()[0]
        print(f"  {n_hit:,} polygons need differencing")

        process_con.execute("""
            CREATE OR REPLACE TABLE canopy_diff AS
            SELECT ST_Difference(ST_MakeValid(c.geom), ST_MakeValid(e.geom)) AS geom
            FROM canopy_hit c
            JOIN excl e ON c.geom && e.geom AND ST_Intersects(c.geom, e.geom)
        """)

        process_con.execute(f"""
            CREATE OR REPLACE TABLE canopy_final AS
            SELECT geom FROM canopy_clear
            UNION ALL
            SELECT geom FROM canopy_diff 
            WHERE NOT ST_IsEmpty(geom) AND ST_Area(geom) >= {min_area}
        """)
        n_excl = process_con.execute("SELECT COUNT(*) FROM canopy_final").fetchone()[0]
        print(f"  {n_excl:,} polygons after OSM exclusion")

    result = process_con.execute("SELECT ST_AsWKB(geom) AS geom FROM canopy_final").fetchdf()
    process_con.close()

    if db_file.exists():
        db_file.unlink()

    from shapely import wkb
    result["geometry"] = result["geom"].apply(lambda b: wkb.loads(bytes(b)))
    gdf = gpd.GeoDataFrame(result.drop(columns=["geom"]), geometry="geometry", crs=CRS_METRIC)
    return gdf


# ─────────────────────────────────────────────────────────────
# Raster Stage Execution
# ─────────────────────────────────────────────────────────────

def run_raster_stage(tile_paths: list, cfg: CityConfig, n_workers: int) -> list:
    worker_args = [
        (str(p), str(cfg.scratch_dir), cfg.boundary, cfg.exg_thresh, cfg.base_area_m2,
         cfg.hue_range, cfg.sat_min, cfg.brightness_gate)
        for p in tile_paths
    ]
    vector_paths = []
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(process_tile, a): a[0] for a in worker_args}
        for f in tqdm(as_completed(futures), total=len(futures), desc=cfg.city):
            tile_path = futures[f]
            try:
                r = f.result(timeout=TILE_TIMEOUT_S)
                if r:
                    vector_paths.append(r)
            except FuturesTimeout:
                tqdm.write(f"[timeout] {Path(tile_path).name}")
                ghost = cfg.scratch_dir / (Path(tile_path).stem + "_vector.parquet")
                ghost.unlink(missing_ok=True)
            except Exception as e:
                tqdm.write(f"[warn] {Path(tile_path).name}: {e}")
    return vector_paths


# ─────────────────────────────────────────────────────────────
# Main Pipeline Entry
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True, help="City key, e.g. 'genova' or 'firenze'")
    p.add_argument("--workers", type=int, default=0, help="Number of parallel workers (0 = auto)")
    args = p.parse_args()

    test_mode = os.environ.get("TEST_MODE", "0") == "1"
    city = args.city.lower()

    print(f"\n{'='*60}")
    print(f"  Canopy extraction (RGB ExG)  |  city={city}  |  test={test_mode}")

    cfg = CityConfig.build(city, test_mode=test_mode)
    print(f"  Tile dir   : {cfg.tile_dir}")
    print(f"  EXG thresh : {cfg.exg_thresh}")
    print(f"  Min area   : {cfg.base_area_m2} m²")
    print(f"  Hue range  : {cfg.hue_range}")
    print(f"  Sat min    : {cfg.sat_min}")
    print(f"  Bright gate: {cfg.brightness_gate}")
    print(f"  Boundary   : {cfg.boundary.geom_type if cfg.boundary else 'none'}")

    tile_paths = [p for p in sorted(cfg.tile_dir.glob("tile_*.tif")) if p.stat().st_size > 10_000]
    if test_mode:
        import random; random.seed(42)
        tile_paths = random.sample(tile_paths, min(10, len(tile_paths)))
        print(f"  [@TEST] {len(tile_paths)} tiles")
    else:
        print(f"  Tiles: {len(tile_paths):,}")

    if not tile_paths:
        print(f"\nNo tiles found in {cfg.tile_dir} matching 'tile_*.tif' (>10KB). Exiting.")
        sys.exit(1)

    with rasterio.open(str(tile_paths[0])) as src:
        first_gsd = abs(src.transform.a)
    print(f"  Index: Excess Green (ExG)  |  GSD: {first_gsd:.3f}m")

    print("\nBuilding OSM exclusion (per-tag cache) ...")
    t0 = time.time()
    base_city = city.replace("_rgb", "")
    exclusion_gdf = build_osm_exclusion(base_city)
    print(f"  Exclusion: {len(exclusion_gdf):,} polygons ({time.time()-t0:.1f}s)" if exclusion_gdf is not None else "  Exclusion: none")
    inclusion_gdf = build_osm_inclusion(base_city)
    print(f"  Inclusion: {len(inclusion_gdf):,} polygons" if inclusion_gdf is not None else "  Inclusion: none")

    existing = [p for p in cfg.scratch_dir.glob("*_vector.parquet") if p.stat().st_size > 1024]
    if len(existing) >= len(tile_paths) * 0.5:
        print(f"\nSkipping raster stage: {len(existing):,} shards cached")
        vector_paths = [str(p) for p in existing]
    else:
        n_workers = args.workers or max(1, os.cpu_count() // 2 - 1)
        print(f"\nRaster stage  ({n_workers} workers) ...")
        t1 = time.time()
        vector_paths = run_raster_stage(tile_paths, cfg, n_workers)
        vector_paths = [str(p) for p in cfg.scratch_dir.glob("*_vector.parquet") if p.stat().st_size > 1024]
        print(f"  {len(vector_paths):,} / {len(tile_paths):,} shards  ({(time.time()-t1)/60:.1f} min)")

    if not vector_paths:
        print("No canopy data. Exiting.")
        sys.exit(0)

    print("\nDuckDB post-processing ...")
    t2 = time.time()
    final = duckdb_postprocess(vector_paths, exclusion_gdf, cfg)
    print(f"  Done in {(time.time()-t2)/60:.1f} min")

    if inclusion_gdf is not None and not inclusion_gdf.empty:
        final = pd.concat([final, inclusion_gdf.to_crs(CRS_METRIC)[["geometry"]]], ignore_index=True)

    out_gpkg = DATA_ROOT / city / "canopy_polygons.gpkg"
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    final.to_crs(CRS_WGS).to_file(str(out_gpkg), driver="GPKG", layer="canopy")
    print(f"\nSaved local archive: {out_gpkg} ({len(final):,} polygons)")

    print(f"\nCommitting results to database environment...")
    try:
        con = connect_db(read_only=False)
        metric_gdf = final.to_crs(CRS_METRIC)
        gdf_to_duckdb(con, metric_gdf, f"preprocessing.{base_city}_canopy")
        con.close()
        print(f"✅ DB committed successfully: preprocessing.{base_city}_canopy")
    except Exception as e:
        print(f"⚠️ Warning: Database extraction transfer failed: {e}")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()