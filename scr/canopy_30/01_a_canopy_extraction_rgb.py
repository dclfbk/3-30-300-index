# canopy_extract_rgb.py — Extract tree canopy from RGB tiles by using excess green (ExG) threshold
# as well as hue/saturation/brightness gates, texture guard, and water guard.

"""
    Usage: 
    1. 
    run 3-30-300-index/notebooks/08_testing_threshold_canopy_ext.py first 
    to check the canopy extraction thresholds for your city
    2.
    in config/config.yaml, set the canopy_extraction block for your city
    3. run 3-30-300-index/scr/canopy_30/01_a_canopy_extraction_rgb.py --city <city>  --n-tiles <number>
       run 3-30-300-index/scr/canopy_30/01_a_canopy_extraction_rgb.py --city <city>
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
from scipy.ndimage import binary_opening, binary_closing

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
from tqdm import tqdm

ROOT_PATH = "/workspace/3-30-300-index"
sys.path.insert(0, ROOT_PATH)
os.chdir(ROOT_PATH)

from config.config import (
    connect_db, gdf_to_duckdb, load_table, get_city_cfg, load_boundary,
    find_tile_dir, build_osm_exclusion, build_osm_inclusion,
    CRS_WGS, CRS_METRIC, DATA_DIR,
    OSM_BUFFER_M, WATER_BUFFER_M, TEST_OSM_BUFFER_DEG,
)



BASE_GSD_M            = 0.20
SIMPLIFY_M            = 1.0
OSM_BUFFER_M          = 2.0
WATER_BUFFER_M        = 6.0
TILE_TIMEOUT_S        = 120 
PRE_SIMPLIFY_M         = 0.15
REDUCE_PRECISION_M     = 0.05
ENABLE_FINAL_SIMPLIFY = True
TEST_OSM_BUFFER_DEG    = 0.001
GRID_CELL_M            = 100
DISSOLVE_STRIP_WIDTH_CELLS = 8
DISSOLVE_THREADS       = 2
NORMAL_THREADS         = 8
NOISE_OPENING_STRUCT = np.ones((2, 2))

# Closing bridges small gaps between nearby true canopy regions (thin
# shadow lines, single-row near-misses between touching crowns).
CANOPY_CLOSING_STRUCT = np.ones((5, 5))

def _tile_footprint_envelope_wgs(tile_paths: list) -> object:
    boxes = []
    crs = None
    for p in tile_paths:
        with rasterio.open(str(p)) as src:
            boxes.append(box(*src.bounds))
            if crs is None:
                crs = src.crs
    envelope = unary_union(boxes)
    envelope_wgs = gpd.GeoSeries([envelope], crs=crs).to_crs(CRS_WGS).iloc[0]
    return envelope_wgs.buffer(TEST_OSM_BUFFER_DEG)


def water_guard_mask(r: np.ndarray, g: np.ndarray, b: np.ndarray,
                      min_green_blue_gap: float = 0.03) -> np.ndarray:
    total = r + g + b
    with np.errstate(invalid="ignore", divide="ignore"):
        g_n = np.where(total > 0, g / total, 0.0)
        b_n = np.where(total > 0, b / total, 0.0)
    return (g_n - b_n) >= min_green_blue_gap

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
    water_guard_gap:  float = 0.0
    texture_min_var:  float = 0.0
    closing_enabled:  bool = False

    @classmethod
    def build(cls, city: str, test_mode: bool = False) -> "CityConfig":
        base_city = city.replace("_rgb", "")

        tile_dir = find_tile_dir(city, base_city)
        scratch_dir = DATA_DIR / f"canopy_{city}_{'test_' if test_mode else ''}tmp"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        boundary = load_boundary(base_city)

        # Per-city canopy-extraction tuning now lives in config.yaml under
        # each city's canopy_extraction: block, not hardcoded dicts here.
        # Tries the full variant name first (e.g. "torino_rgb"), falling
        # back to the base city (e.g. "torino") only if the variant has
        # no config.yaml entry of its own — RGB/NIR variants of the same
        # physical city can have genuinely different tuning without
        # silently missing overrides either way.
        def _canopy_cfg(name):
            try:
                return get_city_cfg(name).get("canopy_extraction", {}) or {}
            except (ValueError, KeyError):
                return {}

        ce = _canopy_cfg(city) or _canopy_cfg(base_city)
        hue_range = ce.get("hue_range")
        if hue_range is not None:
            hue_range = tuple(hue_range)

        return cls(
            city=city,
            tile_dir=tile_dir,
            scratch_dir=scratch_dir,
            db_table=f"preprocessing.canopy_{city}",
            exg_thresh=ce.get("exg_thresh", 0.15),
            base_area_m2=ce.get("min_area_m2", 0.5),
            boundary=boundary,
            hue_range=hue_range,
            sat_min=ce.get("sat_min", 0.0),
            brightness_gate=ce.get("brightness_gate", 35.0),
            water_guard_gap=ce.get("water_guard_gap", 0.0),
            texture_min_var=ce.get("texture_min_var", 0.0),
            closing_enabled=ce.get("closing_enabled", False),
        )

def texture_guard_mask(r: np.ndarray, g: np.ndarray, b: np.ndarray,
                        window: int = 5, min_local_var: float = 0.0008) -> np.ndarray:
    """Reject pixels sitting in an overly SMOOTH/uniform local neighborhood.

    For cities where shadowed real canopy and bright, uniformly-colored
    lawn/meadow genuinely overlap in hue/saturation space (confirmed for
    firenze_rgb — the spectral sampler's own measured-percentile
    suggestion matched the already-configured thresholds almost exactly,
    yet still produced both under-detected shadowed canopy AND
    over-detected bright lawn simultaneously), color alone can't
    separate them. Real tree canopy has natural texture at the
    pixel-neighborhood scale (leaf clumps, branch gaps, light/shadow
    speckle); a mowed lawn or manicured meadow is comparatively smooth
    even when similarly colored. This computes local variance of ExG in
    a small window (fast box-filter approximation, not a slow
    generic_filter) and requires a minimum roughness to pass.

    NOTE: min_local_var is a rough starting guess, not empirically
    derived — sanity-check visually in QGIS and adjust. Too high will
    start rejecting real sparse/thin canopy; too low won't discriminate
    from smooth lawn at all.
    """
    from scipy.ndimage import uniform_filter
    total = r + g + b
    with np.errstate(invalid="ignore", divide="ignore"):
        r_n = np.where(total > 0, r / total, 0.0)
        g_n = np.where(total > 0, g / total, 0.0)
        b_n = np.where(total > 0, b / total, 0.0)
    exg = 2 * g_n - r_n - b_n
    local_mean = uniform_filter(exg, size=window)
    local_sqmean = uniform_filter(exg ** 2, size=window)
    local_var = np.clip(local_sqmean - local_mean ** 2, 0, None)
    return local_var >= min_local_var


def exg_mask(r: np.ndarray, g: np.ndarray, b: np.ndarray, thresh: float,
             brightness_gate: float = 35.0,
             hue_range: Optional[tuple] = None,
             sat_min: float = 0.0,
             water_guard_gap: float = 0.0,
             texture_min_var: float = 0.0) -> np.ndarray:
    total = r + g + b

    brightness_mask = total > brightness_gate
    if not brightness_mask.any():
        return np.zeros(r.shape, dtype=np.uint8)

    with np.errstate(invalid="ignore", divide="ignore"):
        r_n = np.where(total > 0, r / total, 0.0)
        g_n = np.where(total > 0, g / total, 0.0)
        b_n = np.where(total > 0, b / total, 0.0)

    exg = 2 * g_n - r_n - b_n
    mask = (exg > thresh) & brightness_mask
    if water_guard_gap > 0:
        mask = mask & water_guard_mask(r, g, b, min_green_blue_gap=water_guard_gap)
    if texture_min_var > 0 and mask.any():
        mask = mask & texture_guard_mask(r, g, b, min_local_var=texture_min_var)

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
             brightness_gate: float = 35.0, water_guard_gap: float = 0.0,
             texture_min_var: float = 0.0, closing_enabled: bool = False) -> tuple[np.ndarray, float]:
    min_area = base_area_m2 * (gsd_m / BASE_GSD_M) ** 2
    mask = exg_mask(data[0], data[1], data[2], exg_thresh,
                     brightness_gate=brightness_gate, hue_range=hue_range, sat_min=sat_min,
                     water_guard_gap=water_guard_gap, texture_min_var=texture_min_var)

    if mask.any():
        if closing_enabled:
            mask = binary_closing(mask, structure=CANOPY_CLOSING_STRUCT).astype(np.uint8)
        # binary_opening disabled here for this test — see caller note
        # mask = binary_opening(mask, structure=NOISE_OPENING_STRUCT).astype(np.uint8)

    return mask, min_area


def process_tile(args: tuple) -> Optional[str]:
    (tile_path, scratch_dir, boundary, exg_thresh, base_area_m2,
     hue_range, sat_min, brightness_gate, water_guard_gap, texture_min_var, closing_enabled) = args
    p = Path(tile_path)

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
                               hue_range=hue_range, sat_min=sat_min, brightness_gate=brightness_gate,
                               water_guard_gap=water_guard_gap, texture_min_var=texture_min_var,
                               closing_enabled=closing_enabled)
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

    if gdf.empty:
        return None

    gdf["geometry"] = gdf.geometry.simplify(PRE_SIMPLIFY_M, preserve_topology=True)
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if gdf.empty:
        return None

    gdf.to_parquet(str(out_parquet))
    return str(out_parquet)


def duckdb_postprocess(vector_paths, exclusion_gdf, cfg: CityConfig):
    min_area = cfg.base_area_m2

    db_file = cfg.scratch_dir / "duckdb_temp.db"
    if db_file.exists():
        db_file.unlink()

    process_con = duckdb.connect(str(db_file))
    process_con.execute("INSTALL spatial; LOAD spatial;")

    process_con.execute(f"""
        SET threads={NORMAL_THREADS};
        SET preserve_insertion_order=false;
        SET memory_limit='5GB';
        SET temp_directory='{cfg.scratch_dir}';
    """)

    print(f"DuckDB: Natively scanning {len(vector_paths):,} Parquet shards ...")

    t_scan = time.time()
    search_pattern = str(cfg.scratch_dir / "*_vector.parquet")
    process_con.execute(f"""
        CREATE OR REPLACE TABLE canopy_raw AS 
        SELECT geometry::GEOMETRY AS geom 
        FROM read_parquet('{search_pattern}') 
        WHERE ST_Area(geometry::GEOMETRY) >= {min_area}
    """)
    n_raw = process_con.execute("SELECT COUNT(*) FROM canopy_raw").fetchone()[0]
    print(f"  {n_raw:,} raw polygons scanned ({time.time()-t_scan:.1f}s)")

    process_con.execute(f"SET threads={DISSOLVE_THREADS}")

    process_con.execute(f"""
        ALTER TABLE canopy_raw ADD COLUMN gx BIGINT;
        ALTER TABLE canopy_raw ADD COLUMN gy BIGINT;
        UPDATE canopy_raw SET
            gx = CAST(FLOOR(ST_X(ST_Centroid(geom)) / {GRID_CELL_M}) AS BIGINT),
            gy = CAST(FLOOR(ST_Y(ST_Centroid(geom)) / {GRID_CELL_M}) AS BIGINT);
    """)
    gx_min, gx_max = process_con.execute("SELECT MIN(gx), MAX(gx) FROM canopy_raw").fetchone()
    n_strips = ((gx_max - gx_min) // DISSOLVE_STRIP_WIDTH_CELLS) + 1

    print(f"Dissolving touching/adjacent polygons "
          f"(grid-partitioned, {GRID_CELL_M}m cells, {n_strips} strips, threads={DISSOLVE_THREADS}) ...")
    t_dissolve = time.time()

    process_con.execute("CREATE OR REPLACE TABLE canopy_dissolved (geom GEOMETRY)")
    for i, strip_start in enumerate(range(gx_min, gx_max + 1, DISSOLVE_STRIP_WIDTH_CELLS)):
        strip_end = strip_start + DISSOLVE_STRIP_WIDTH_CELLS - 1
        process_con.execute(f"""
            INSERT INTO canopy_dissolved
            SELECT UNNEST(ST_Dump(ST_Union_Agg(ST_ReducePrecision(geom, {REDUCE_PRECISION_M})))).geom AS geom
            FROM canopy_raw
            WHERE gx BETWEEN {strip_start} AND {strip_end}
            GROUP BY gx, gy
        """)
        if (i + 1) % 5 == 0 or strip_end >= gx_max:
            n_so_far = process_con.execute("SELECT COUNT(*) FROM canopy_dissolved").fetchone()[0]
            print(f"    strip {i+1}/{n_strips}  ({n_so_far:,} polygons so far, "
                  f"{time.time()-t_dissolve:.0f}s elapsed)")

    n_dissolved = process_con.execute("SELECT COUNT(*) FROM canopy_dissolved").fetchone()[0]
    print(f"  {n_dissolved:,} polygons after dissolve ({time.time()-t_dissolve:.1f}s, was {n_raw:,})")
    process_con.execute(f"SET threads={NORMAL_THREADS}")

    if ENABLE_FINAL_SIMPLIFY:
        print("Simplifying ...")
        t_simplify = time.time()
        process_con.execute(f"""
            CREATE OR REPLACE TABLE canopy_final AS
            SELECT ST_SimplifyPreserveTopology(geom, {SIMPLIFY_M})::GEOMETRY AS geom
            FROM canopy_dissolved
            WHERE ST_Area(geom) >= {min_area}
        """)
        n_final = process_con.execute("SELECT COUNT(*) FROM canopy_final").fetchone()[0]
        print(f"  {n_final:,} polygons after simplify ({time.time()-t_simplify:.1f}s)")
    else:
        print("Skipping final simplify (ENABLE_FINAL_SIMPLIFY=False) ...")
        process_con.execute(f"""
            CREATE OR REPLACE TABLE canopy_final AS
            SELECT geom::GEOMETRY AS geom
            FROM canopy_dissolved
            WHERE ST_Area(geom) >= {min_area}
        """)
        n_final = process_con.execute("SELECT COUNT(*) FROM canopy_final").fetchone()[0]
        print(f"  {n_final:,} polygons (unsimplified, dissolve-only)")

    if exclusion_gdf is not None:
        print("Applying OSM exclusion via DuckDB (filter-then-diff) ...")
        t_excl = time.time()
        excl_path = str(cfg.scratch_dir / "osm_exclusion_tmp.parquet")
        exclusion_gdf.to_parquet(excl_path)

        process_con.execute(f"""
            CREATE OR REPLACE TABLE excl AS
            SELECT geometry::GEOMETRY AS geom FROM read_parquet('{excl_path}')
        """)

        process_con.execute("""
            CREATE OR REPLACE TABLE canopy_final AS
            SELECT geom::GEOMETRY AS geom FROM canopy_final
        """)

        process_con.execute("CREATE INDEX excl_rtree ON excl USING RTREE(geom)")
        process_con.execute("CREATE INDEX canopy_rtree ON canopy_final USING RTREE(geom)")

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
        print(f"  {n_excl:,} polygons after OSM exclusion ({time.time()-t_excl:.1f}s)")

    t_fetch = time.time()
    result = process_con.execute("SELECT ST_AsWKB(geom) AS geom FROM canopy_final").fetchdf()
    print(f"  Fetched {len(result):,} final rows ({time.time()-t_fetch:.1f}s)")
    process_con.close()

    if db_file.exists():
        db_file.unlink()

    from shapely import wkb
    result["geometry"] = result["geom"].apply(lambda b: wkb.loads(bytes(b)))
    gdf = gpd.GeoDataFrame(result.drop(columns=["geom"]), geometry="geometry", crs=CRS_METRIC)
    return gdf


def run_raster_stage(tile_paths: list, cfg: CityConfig, n_workers: int) -> list:
    worker_args = [
        (str(p), str(cfg.scratch_dir), cfg.boundary, cfg.exg_thresh, cfg.base_area_m2,
         cfg.hue_range, cfg.sat_min, cfg.brightness_gate, cfg.water_guard_gap, cfg.texture_min_var,
         cfg.closing_enabled)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True, help="City key, e.g. 'genova' or 'firenze'")
    p.add_argument("--workers", type=int, default=0, help="Number of parallel workers (0 = auto)")
    p.add_argument("--n-tiles", type=int, default=None,
                   help="If set, sample this many random tiles for a quick test run "
                        "instead of processing the whole city. Overrides TEST_MODE=1 "
                        "env var's fixed count of 10 if both are set.")
    p.add_argument("--skip-osm", action="store_true",
                   help="Skip OSM exclusion/inclusion fetch entirely for fast "
                        "iteration when testing thresholds. Output will include "
                        "buildings/grass/water etc. unfiltered — for spectral "
                        "threshold tuning only, not for a real extraction run.")
    args = p.parse_args()

    test_mode_env = os.environ.get("TEST_MODE", "0") == "1"
    n_tiles = args.n_tiles if args.n_tiles is not None else (10 if test_mode_env else None)
    test_mode = n_tiles is not None
    city = args.city.lower()

    print(f"\n{'='*60}")
    print(f"  Canopy extraction (RGB ExG)  |  city={city}  |  test={test_mode}"
          f"{f' (n_tiles={n_tiles})' if test_mode else ''}"
          f"{'  |  SKIP-OSM' if args.skip_osm else ''}")

    cfg = CityConfig.build(city, test_mode=test_mode)
    print(f"  Tile dir   : {cfg.tile_dir}")
    print(f"  EXG thresh : {cfg.exg_thresh}")
    print(f"  Min area   : {cfg.base_area_m2} m²")
    print(f"  Hue range  : {cfg.hue_range}")
    print(f"  Sat min    : {cfg.sat_min}")
    print(f"  Bright gate: {cfg.brightness_gate}")
    print(f"  Water guard: {cfg.water_guard_gap if cfg.water_guard_gap > 0 else 'off'}")
    print(f"  Texture guard: {cfg.texture_min_var if cfg.texture_min_var > 0 else 'off (experimental)'}")
    print(f"  Closing (gap-bridging): {'ON' if cfg.closing_enabled else 'off'}")
    print(f"  Simplify   : {SIMPLIFY_M} m (final simplify {'ON' if ENABLE_FINAL_SIMPLIFY else 'OFF'})")
    print(f"  Boundary   : {cfg.boundary.geom_type if cfg.boundary else 'none'}")

    tile_paths = [p for p in sorted(cfg.tile_dir.glob("tile_*.tif")) if p.stat().st_size > 10_000]
    if test_mode:
        import random; random.seed(42)
        tile_paths = random.sample(tile_paths, min(n_tiles, len(tile_paths)))
        print(f"  [@TEST] {len(tile_paths)} tiles")
    else:
        print(f"  Tiles: {len(tile_paths):,}")

    if not tile_paths:
        print(f"\nNo tiles found in {cfg.tile_dir} matching 'tile_*.tif' (>10KB). Exiting.")
        sys.exit(1)

    with rasterio.open(str(tile_paths[0])) as src:
        first_gsd = abs(src.transform.a)
    print(f"  Index: Excess Green (ExG)  |  GSD: {first_gsd:.3f}m")

    base_city = city.replace("_rgb", "")

    if args.skip_osm:
        print(f"\n[SKIP-OSM] OSM exclusion/inclusion fetch skipped for fast testing — "
              f"output will NOT have buildings/grass/water/etc. filtered out.")
        exclusion_gdf = None
        inclusion_gdf = None
    else:
        if test_mode:
            query_geom = _tile_footprint_envelope_wgs(tile_paths)
            osm_scope = "test"
        else:
            query_geom = gpd.GeoSeries([cfg.boundary], crs=CRS_METRIC).to_crs(CRS_WGS).iloc[0]
            osm_scope = "full"

        print(f"\nBuilding OSM exclusion (per-tag cache, scope={osm_scope}) ...")
        t0 = time.time()
        exclusion_gdf = build_osm_exclusion(base_city, query_geom, osm_scope)
        print(f"  Exclusion: {len(exclusion_gdf):,} polygons ({time.time()-t0:.1f}s)" if exclusion_gdf is not None else "  Exclusion: none")
        inclusion_gdf = build_osm_inclusion(base_city, query_geom, osm_scope)
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
        inclusion_metric = inclusion_gdf.to_crs(CRS_METRIC)[["geometry"]]
        t_trim = time.time()
        inclusion_trimmed = gpd.overlay(inclusion_metric, final[["geometry"]],
                                         how="difference", keep_geom_type=False)
        inclusion_trimmed = inclusion_trimmed[~inclusion_trimmed.geometry.is_empty].copy()
        print(f"  Inclusion trimmed against ExG-detected canopy: "
              f"{len(inclusion_metric):,} -> {len(inclusion_trimmed):,} non-overlapping polygons "
              f"({time.time()-t_trim:.1f}s)")
        final = pd.concat([final, inclusion_trimmed], ignore_index=True)

    out_gpkg = DATA_DIR / city / "canopy_polygons.gpkg"
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    final.to_crs(CRS_WGS).to_file(str(out_gpkg), driver="GPKG", layer="canopy")
    print(f"\nSaved local archive: {out_gpkg} ({len(final):,} polygons)")

    print(f"\nCommitting results to database environment...")
    try:
        con = connect_db(read_only=False)
        metric_gdf = final.to_crs(CRS_METRIC)
        gdf_to_duckdb(con, metric_gdf, f"preprocessing.{base_city}_canopy")
        con.close()
        print(f"DB committed successfully: preprocessing.{base_city}_canopy")
    except Exception as e:
        print(f"Warning: Database extraction transfer failed: {e}")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()