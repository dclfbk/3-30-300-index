"""
canopy_extract_nir.py — Extract tree canopy from NIR/multispectral tiles.
uses NDVI (NIR-based) instead of ExG (RGB-based) for vegetation detection.

Usage:
    python canopy_extract_nir.py --city bologna_nir
    TEST_MODE=1 python canopy_extract_nir.py --city firenze_cir --boundary-city firenze
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

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, box
from skimage.filters import threshold_otsu
import duckdb
from tqdm import tqdm

from config.config import (
    connect_db, gdf_to_duckdb, get_city_cfg, load_boundary, find_tile_dir,
    build_osm_exclusion, build_osm_inclusion,
    CRS_WGS, CRS_METRIC, DATA_DIR, ROOT_PATH,
)

sys.path.insert(0, ROOT_PATH)
os.chdir(ROOT_PATH)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

BASE_GSD_M = 0.20
NDVI_THRESH_FALLBACK  = 0.3
SIMPLIFY_M  = 0.25  # Preserve tree line geometry better than RGB
TILE_TIMEOUT_S = 120

# Per-city NDVI thresholds — fallback only. Prefer setting a per-city
# `ndvi_extraction:` block in config.yaml (same pattern as canopy_extraction
# for the RGB pipeline); these dicts are used only when config.yaml has no
# entry for the city yet.
CITY_NDVI_THRESH_FALLBACK = {
    "bolzano": 0.40, "firenze": 0.28, "bologna": 0.30, "torino":  0.30,
    "verona":  0.28, "venezia": 0.32, "milano":  0.28, "roma":    0.25,
    "bari":    0.20, "trento":  0.28, "perugia": 0.28,
}
CITY_MIN_AREA_M2_FALLBACK = {
    "bolzano": 1.5, "firenze": 1.5, "bologna": 1.5, "torino":  1.5,
    "verona":  2.0, "venezia": 2.0, "milano":  1.5, "roma":    1.5,
    "trento":  1.0, "perugia": 1.0, "bari":    1.0,
}


# ─────────────────────────────────────────────────────────────
# City config dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class CityConfig:
    city:         str
    tile_dir:     Path
    scratch_dir:  Path
    db_table:     str
    ndvi_thresh:  float
    base_area_m2: float
    is_cir:       bool = False
    boundary:     object = field(default=None, repr=False)  # shapely geometry

    @classmethod
    def build(cls, city: str, test_mode: bool = False,
              boundary_city: Optional[str] = None,
              boundary_path: Optional[str] = None,
              force_cir: bool = False) -> "CityConfig":

        base_city = city.replace("_rgb", "").replace("_nir", "").replace("_cir", "")
        target_boundary_city = boundary_city if boundary_city else base_city

        tile_dir = find_tile_dir(city, base_city)
        scratch_dir = DATA_DIR / f"canopy_{city}_{'test_' if test_mode else ''}tmp"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        boundary = load_boundary(target_boundary_city, boundary_path)

        def _ndvi_cfg(name):
            try:
                return get_city_cfg(name).get("ndvi_extraction", {}) or {}
            except (ValueError, KeyError):
                return {}

        ne = _ndvi_cfg(city) or _ndvi_cfg(base_city)
        ndvi_thresh = ne.get(
            "ndvi_thresh",
            CITY_NDVI_THRESH_FALLBACK.get(base_city, NDVI_THRESH_FALLBACK),
        )
        base_area_m2 = ne.get(
            "min_area_m2",
            CITY_MIN_AREA_M2_FALLBACK.get(base_city, 1.0),
        )

        # CIR band order (NIR/R/G instead of standard RGB/multispectral) is
        # only inferred from an explicit "_cir" suffix or --force-cir, never
        # hardcoded to a specific city name — add "_cir" to the city key in
        # config.yaml/CLI for any future CIR source.
        is_cir = force_cir or city.lower().endswith("_cir")

        return cls(
            city=city,
            tile_dir=tile_dir,
            scratch_dir=scratch_dir,
            db_table=f"preprocessing.canopy_{city}",
            ndvi_thresh=ndvi_thresh,
            base_area_m2=base_area_m2,
            is_cir=is_cir,
            boundary=boundary,
        )


# ─────────────────────────────────────────────────────────────
# Band detection & vegetation indices
# ─────────────────────────────────────────────────────────────

def detect_nir_red(data: np.ndarray, is_cir: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Extract NIR and Red bands from multispectral stack."""
    n = data.shape[0]

    if is_cir:
        # CIR (NIR/R/G) -> Band 1 is NIR, Band 2 is Red
        return data[0], data[1]

    if n >= 4:
        if np.mean(data[3]) > np.mean(data[0]) * 1.2:
            return data[3], data[0]  # Band 4 = NIR, Band 1 = Red
        return data[0], data[1]      # Band 1 = NIR, Band 2 = Red

    if n == 3:
        return data[0], data[1]

    return data[0], data[0] if n > 0 else (np.zeros((1, 1)), np.zeros((1, 1)))


def _otsu_ndvi_thresh(nir: np.ndarray, red: np.ndarray) -> float:
    """Compute adaptive NDVI threshold via Otsu's method."""
    denom = nir + red
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)
    valid = ndvi[np.isfinite(ndvi)].ravel()
    if len(valid) < 100 or valid.std() < 0.01:
        return NDVI_THRESH_FALLBACK
    try:
        return float(np.clip(threshold_otsu(valid), 0.1, 0.6))
    except Exception:
        return NDVI_THRESH_FALLBACK


def ndvi_mask(nir: np.ndarray, red: np.ndarray, user_thresh: float) -> np.ndarray:
    denom = nir + red
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)

    # user_thresh (from config.yaml / fallback dict) takes priority;
    # Otsu is only a backstop when no per-city value is configured at all.
    final_thresh = user_thresh if user_thresh is not None else _otsu_ndvi_thresh(nir, red)

    return (ndvi > final_thresh).astype(np.uint8)


def veg_mask(data: np.ndarray, gsd_m: float, ndvi_thresh: float,
             base_area_m2: float, is_cir: bool = False) -> tuple[np.ndarray, float]:
    """Return (uint8 NDVI mask, effective min area m²) scaled by GSD."""
    min_area = base_area_m2 * (gsd_m / BASE_GSD_M) ** 2
    nir, red = detect_nir_red(data, is_cir=is_cir)
    mask = ndvi_mask(nir, red, ndvi_thresh)
    return mask, min_area


# ─────────────────────────────────────────────────────────────
# Per-tile worker (pure function — no globals)
# ─────────────────────────────────────────────────────────────
def process_tile(args: tuple) -> Optional[str]:
    """Process one NIR tile: vectorize NDVI mask → GeoPackage shard."""
    tile_path, scratch_dir, boundary, ndvi_thresh, base_area_m2, is_cir = args
    p = Path(tile_path)
    out_gpkg = Path(scratch_dir) / (p.stem + "_vector.gpkg")

    if out_gpkg.exists() and out_gpkg.stat().st_size > 4096:
        return str(out_gpkg)

    try:
        with rasterio.open(tile_path) as src:
            n_bands = min(src.count, 4)
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

    mask, min_area = veg_mask(data, gsd_m, ndvi_thresh, base_area_m2, is_cir=is_cir)
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
        gdf.to_file(str(out_gpkg), driver="GPKG", layer="canopy")
        return str(out_gpkg)
    return None


# ─────────────────────────────────────────────────────────────
# DuckDB post-processing
# ─────────────────────────────────────────────────────────────

def duckdb_postprocess(vector_paths: list, exclusion_gdf: Optional[gpd.GeoDataFrame],
                       scratch_dir: Path, base_area_m2: float) -> gpd.GeoDataFrame:
    """Merge shard GeoPackages, simplify, apply OSM exclusion, return unified GeoDataFrame."""
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("SET max_memory='6GB';")
    con.execute("SET preserve_insertion_order=false;")

    valid = [
        str(Path(p).resolve())
        for p in vector_paths
        if Path(p).exists() and Path(p).stat().st_size > 4096
    ]
    print(f"  DuckDB: loading {len(valid):,} shards ...")

    union_sql = " UNION ALL ".join(
        f"SELECT geom FROM ST_Read('{p}') WHERE ST_Area(geom) >= {base_area_m2}"
        for p in valid
    )
    con.execute(f"CREATE OR REPLACE TABLE canopy_raw AS {union_sql}")
    n_raw = con.execute("SELECT COUNT(*) FROM canopy_raw").fetchone()[0]
    print(f"    {n_raw:,} raw polygons")

    con.execute(f"""
        CREATE OR REPLACE TABLE canopy_final AS
        SELECT ST_SimplifyPreserveTopology(geom, {SIMPLIFY_M}) AS geom
        FROM canopy_raw WHERE ST_Area(geom) >= {base_area_m2}
    """)
    n_simp = con.execute("SELECT COUNT(*) FROM canopy_final").fetchone()[0]
    print(f"    {n_simp:,} after simplification")

    if exclusion_gdf is not None and not exclusion_gdf.empty:
        print(f"    Applying OSM exclusion (indexed spatial join) ...")

        excl_path = str(scratch_dir / "osm_exclusion_features.gpkg")
        exclusion_gdf[["geometry"]].to_file(excl_path, driver="GPKG")

        con.execute(f"CREATE OR REPLACE TABLE excl AS SELECT geom FROM ST_Read('{excl_path}')")

        con.execute("""
            CREATE OR REPLACE TABLE canopy_intersected AS
            SELECT 
                c.rowid as c_id,
                ST_Difference(ST_MakeValid(c.geom), ST_CollectionExtract(ST_Union_Agg(ST_MakeValid(e.geom)), 3)) AS geom
            FROM canopy_final c
            JOIN excl e ON ST_Intersects(c.geom, e.geom)
            GROUP BY c.rowid, c.geom
        """)

        con.execute(f"""
            CREATE OR REPLACE TABLE canopy_final AS
            SELECT geom FROM canopy_intersected WHERE NOT ST_IsEmpty(geom) AND ST_Area(geom) >= {base_area_m2}
            UNION ALL
            SELECT geom FROM canopy_final 
            WHERE rowid NOT IN (SELECT c_id FROM canopy_intersected)
        """)
        n_excl = con.execute("SELECT COUNT(*) FROM canopy_final").fetchone()[0]
        print(f"    {n_excl:,} after OSM exclusion")

    result = con.execute("SELECT ST_AsWKB(geom) AS geom FROM canopy_final").fetchdf()
    con.close()

    from shapely import wkb
    result["geometry"] = result["geom"].apply(lambda b: wkb.loads(bytes(b)))
    return gpd.GeoDataFrame(result.drop(columns=["geom"]), geometry="geometry", crs=CRS_METRIC)


# ─────────────────────────────────────────────────────────────
# Raster stage orchestration
# ─────────────────────────────────────────────────────────────
def run_raster_stage(tile_paths: list, cfg: CityConfig, n_workers: int) -> list:
    """Parallelize tile processing with progress bar."""
    worker_args = [
        (str(p), str(cfg.scratch_dir), cfg.boundary, cfg.ndvi_thresh, cfg.base_area_m2, cfg.is_cir)
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
                ghost = cfg.scratch_dir / (Path(tile_path).stem + "_vector.gpkg")
                ghost.unlink(missing_ok=True)
            except Exception as e:
                tqdm.write(f"[warn] {Path(tile_path).name}: {e}")
    return vector_paths


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True, help="City key, e.g. 'bari' or 'firenze_cir'")
    p.add_argument("--workers", type=int, default=0, help="Number of parallel workers (0 = auto)")
    p.add_argument("--boundary-city", default=None,
                    help="Boundary table to use if different from --city (e.g. base city for *_rgb/_nir/_cir variants)")
    p.add_argument("--boundary-path", default=None,
                    help="Load boundary from a file (shp/gpkg/geojson) instead of the DB")
    p.add_argument("--force-cir", action="store_true",
                    help="Force CIR band order (NIR/R/G) even if city key doesn't end in '_cir'")
    args = p.parse_args()

    test_mode = os.environ.get("TEST_MODE", "0") == "1"
    city = args.city.lower()

    print(f"\n{'='*60}")
    print(f"  Canopy extraction (NIR)  |  city={city}  |  test={test_mode}")

    cfg = CityConfig.build(
        city=city,
        test_mode=test_mode,
        boundary_city=args.boundary_city,
        boundary_path=args.boundary_path,
        force_cir=args.force_cir,
    )
    print(f"  Tile dir      : {cfg.tile_dir}")
    print(f"  NDVI thresh   : {cfg.ndvi_thresh:.2f}")
    print(f"  Min area      : {cfg.base_area_m2} m²")
    print(f"  CIR band order: {'yes' if cfg.is_cir else 'no'}")
    print(f"  Boundary      : {cfg.boundary.geom_type if cfg.boundary else 'none'}")

    tile_paths = [p for p in sorted(cfg.tile_dir.glob("tile_*.tif"))
                  if p.stat().st_size > 10_000]
    if test_mode:
        import random
        random.seed(42)
        tile_paths = random.sample(tile_paths, min(10, len(tile_paths)))
        print(f"  [@TEST] {len(tile_paths)} tiles (sampled)")
    else:
        print(f"  Tiles: {len(tile_paths):,}")

    if not tile_paths:
        print("  ERROR: No tiles found.")
        sys.exit(1)

    with rasterio.open(str(tile_paths[0])) as src:
        first_gsd = abs(src.transform.a)
    print(f"  Index: NDVI  |  GSD: {first_gsd:.3f}m")

    base_city = city.replace("_rgb", "").replace("_nir", "").replace("_cir", "")

    # OSM — same shared exclusion/inclusion tag sets and polygon+scope fetch
    # strategy as the RGB pipeline (config.py: build_osm_exclusion/
    # build_osm_inclusion), instead of NIR's previous separate, narrower
    # tag lists and place-name query.
    osm_scope = "test" if test_mode else "full"
    query_geom = gpd.GeoSeries([cfg.boundary], crs=CRS_METRIC).to_crs(CRS_WGS).iloc[0]

    print(f"\nBuilding OSM exclusion (per-tag cache, scope={osm_scope}) ...")
    t0 = time.time()
    exclusion_gdf = build_osm_exclusion(base_city, query_geom, osm_scope)
    print(f"  Exclusion: {len(exclusion_gdf):,} polygons ({time.time()-t0:.1f}s)"
          if exclusion_gdf is not None else "  Exclusion: none")
    inclusion_gdf = build_osm_inclusion(base_city, query_geom, osm_scope)
    print(f"  Inclusion: {len(inclusion_gdf):,} polygons"
          if inclusion_gdf is not None else "  Inclusion: none")

    existing = [p for p in cfg.scratch_dir.glob("*_vector.gpkg") if p.stat().st_size > 4096]
    if len(existing) >= len(tile_paths) * 0.5:
        print(f"\nSkipping raster stage: {len(existing):,} shards cached")
        vector_paths = [str(p) for p in existing]
    else:
        n_workers = args.workers or max(1, os.cpu_count() // 2 - 1)
        print(f"\nRaster stage ({n_workers} workers) ...")
        t1 = time.time()
        run_raster_stage(tile_paths, cfg, n_workers)
        vector_paths = [str(p) for p in cfg.scratch_dir.glob("*_vector.gpkg") if p.stat().st_size > 4096]
        print(f"  {len(vector_paths):,} / {len(tile_paths):,} shards  ({(time.time()-t1)/60:.1f} min)")

    if not vector_paths:
        print("No canopy data found. Exiting.")
        sys.exit(0)

    print("\nDuckDB post-processing ...")
    t2 = time.time()
    final = duckdb_postprocess(vector_paths, exclusion_gdf, cfg.scratch_dir, cfg.base_area_m2)
    print(f"  Done in {(time.time()-t2)/60:.1f} min")

    if inclusion_gdf is not None and not inclusion_gdf.empty:
        final = pd.concat(
            [final, inclusion_gdf.to_crs(CRS_METRIC)[["geometry"]]],
            ignore_index=True
        )

    out_gpkg = DATA_DIR / city / "canopy_polygons.gpkg"
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    final.to_crs(CRS_WGS).to_file(str(out_gpkg), driver="GPKG", layer="canopy")
    print(f"\nSaved: {out_gpkg} ({len(final):,} polygons)")

    print(f"\nCommitting to database ...")
    try:
        con = connect_db(read_only=False)
        metric_gdf = final.to_crs(CRS_METRIC)
        gdf_to_duckdb(con, metric_gdf, f"preprocessing.{base_city}_canopy_nir")
        con.close()
        print(f" DB committed: preprocessing.{base_city}_canopy_nir")
    except Exception as e:
        print(f" DB commit failed: {e}")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()