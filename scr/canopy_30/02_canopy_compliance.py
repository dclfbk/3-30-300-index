#compute compliance of 30% per h3 cell, and save to db
#compliance means = if the canopy cover is >= 30% of the total area of the h3 cell
#outputs: 
#1. compliance per h3
#2. compliance per civic numbers per h3 (how many civic numbers live in compliant h3 cells)
#3. summary results written in /data/rule30_compliance_summary.csv

"""
Usage:
    run 3-30-300-index/scr/canopy_30/02_canopy_compliance.py --city <city>  
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import logging
import multiprocessing
import warnings
warnings.filterwarnings("ignore")
import osmnx as ox

import numpy as np
import pandas as pd
import geopandas as gpd
import h3
import duckdb
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from shapely.geometry import Polygon, box
from shapely.validation import make_valid

ROOT_PATH = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_PATH))
os.chdir(ROOT_PATH)

from config.config import (
    connect_db_with_retry, load_table, gdf_to_duckdb, get_city_cfg, strip_source_suffix,
    _osm_cache_path, _fetch_osm_tag, connect_db_with_retry,
    CRS_WGS, CRS_METRIC, DATA_DIR, WATER_TAG_KEYS,
    DB_PATH, DB_LOCK_MAX_RETRIES, DB_LOCK_BASE_WAIT_S, DB_LOCK_MAX_WAIT_S,
)




logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)
SUMMARY_CSV = DATA_DIR / "rule30_compliance_summary.csv"
RES_HEX = 8              # H3 resolution 8 ~ 0.6 km^2 hexagons — 3-30-300 handbook guidance
CANOPY_THRESHOLD = 30.0  # 30% canopy cover


def load_or_fetch_water(con, city: str, boundary_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Attempts to load water polygons from existing schemas/tables.
    If not found, fetches from OSM (per-tag, cached via config.py's
    fetch_osm_tag) and caches the combined result into DuckDB.
    """
    base_city = strip_source_suffix(city)
    try:
        water = load_table(con, "raw", f"{base_city}_water")
        print(f"[water] Found primary raw.{base_city}_water in DB.")
        return water.to_crs(CRS_METRIC)
    except Exception:
        pass

    alternative_tables = [f"water_{city}", f"water_mask_{city}", f"excluded_water_{city}"]
    for schema in ["preprocessing", "public"]:
        for table in alternative_tables:
            try:
                water = load_table(con, schema, table)
                print(f"[water] Found cached water mask in alternative table: {schema}.{table}")
                gdf_to_duckdb(con, water, f"raw.{city}_water")
                return water.to_crs(CRS_METRIC)
            except Exception:
                continue

    print(f"[water] {city}_water not found in DuckDB. Fetching from OSM (per-tag cache)...")
    boundary_wgs = boundary_gdf.to_crs(CRS_WGS)
    polygon = boundary_wgs.geometry.unary_union

    gdfs = []
    for key, values in WATER_TAG_KEYS.items():
        for value in values:
            gdf = _fetch_osm_tag(base_city, key, value, polygon, scope="full")
            if gdf is not None:
                gdfs.append(gdf)

    if not gdfs:
        print(f"[water] WARNING: No water bodies detected inside {city} boundary.")
        water_gdf = gpd.GeoDataFrame(columns=["geometry"], crs=CRS_WGS)
    else:
        water_gdf = gpd.GeoDataFrame(
            pd.concat(gdfs, ignore_index=True), crs=CRS_WGS
        )[["geometry"]].reset_index(drop=True)
        print(f"[water] Successfully downloaded {len(water_gdf)} water features from OSM.")

    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    gdf_to_duckdb(con, water_gdf, f"raw.{city}_water")
    print(f"[water] Saved fetched data to -> raw.{city}_water")

    return water_gdf.to_crs(CRS_METRIC)


def finite_geom(geom) -> bool:
    """Reject geometries with non-finite coordinates (NaN/Inf) that break spatial ops."""
    return np.isfinite(np.array(geom.bounds)).all()


def h3_to_gdf(cells: list[str]) -> gpd.GeoDataFrame:
    """Convert a list of H3 cell IDs to a GeoDataFrame of polygons in CRS_METRIC."""
    polys = [Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in cells]
    gdf = gpd.GeoDataFrame({"h3_cell": cells}, geometry=polys, crs=CRS_WGS).to_crs(CRS_METRIC)
    return gdf[gdf.geometry.apply(finite_geom)].reset_index(drop=True)


def build_hex_layer(con, city: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build the H3 hex grid for the city."""
    base_city = strip_source_suffix(city)
    civici   = load_table(con, "raw", f"{base_city}_buildings_sn")
    boundary = load_table(con, "raw", f"{base_city}_boundary")
    boundary_geom = boundary.geometry.iloc[0]

    civici_wgs = civici.to_crs(CRS_WGS)
    civici_wgs["h3_cell"] = civici_wgs.geometry.apply(
        lambda g: h3.latlng_to_cell(g.y, g.x, RES_HEX)
    )

    civic_counts = civici_wgs.groupby("h3_cell").size().rename("civic_count")
    tiles = h3_to_gdf(civic_counts.index.tolist())
    tiles["civic_count"] = tiles["h3_cell"].map(civic_counts).fillna(0).astype(int)

    tiles = tiles[tiles.geometry.intersects(boundary_geom)].copy()
    tiles["geometry"] = tiles.geometry.intersection(boundary_geom)
    tiles = tiles[~tiles.geometry.is_empty & tiles.geometry.apply(finite_geom)].reset_index(drop=True)

    log.info("[%s] Hex tiles after boundary clip: %d", city, len(tiles))

    con.execute("CREATE SCHEMA IF NOT EXISTS preprocessing")
    gdf_to_duckdb(con, tiles, f"preprocessing.hex_grid_{city}")
    log.info("[%s] Intermediate hex grid saved -> preprocessing.hex_grid_%s", city, city)

    return tiles, civici_wgs


def _load_and_prepare_canopy(con, city: str) -> gpd.GeoDataFrame:
    """Load canopy polygons, preferring the DB cache but falling back to
    canopy_testing.py's own gpkg output if the DB table is missing."""
    table_name = f"{city}_canopy"
    try:
        canopy = load_table(con, "preprocessing", table_name)
        log.info("[%s] preprocessing.%s found in DB (%d rows) — using it",
                  city, table_name, len(canopy))
        return canopy
    except Exception:
        pass

    gpkg_path = DATA_DIR / city / "canopy_polygons.gpkg"
    if not gpkg_path.exists():
        raise FileNotFoundError(
            f"No canopy data found for '{city}': preprocessing.{table_name} "
            f"doesn't exist in the DB, and {gpkg_path} doesn't exist either. "
            f"Run canopy_testing.py --city {city} first (a full, non-test run)."
        )

    log.info("[%s] preprocessing.%s not in DB — loading from %s instead",
              city, table_name, gpkg_path)
    canopy = gpd.read_file(gpkg_path, layer="canopy")

    con.execute("CREATE SCHEMA IF NOT EXISTS preprocessing")
    gdf_to_duckdb(con, canopy.to_crs(CRS_METRIC), f"preprocessing.{table_name}")
    log.info("[%s] Saved -> preprocessing.%s (%d rows)", city, table_name, len(canopy))

    return canopy


def _prefilter_validate_clip(gdf: gpd.GeoDataFrame, clip_geom, simplify_m: float = 0.0,
                              label: str = "") -> gpd.GeoDataFrame:
    """FIX: replaces the old make_valid() -> filter -> clip order, which
    ran the expensive make_valid() repair on the ENTIRE city-wide
    dataset before discarding anything geographically irrelevant. For a
    large city (Genova's canopy table has run past 800K rows; Milano's
    past 290K), that made make_valid() -- a genuine per-geometry repair
    operation, run via a slow row-by-row .apply() -- the dominant,
    previously-unmeasured cost in this function, silently undoing the
    benefit of every later clip-before-overlay optimization.

    New order: cheap spatial-index bbox pre-filter FIRST (near-free --
    no geometry repair needed for an intersects() bbox check), THEN
    make_valid() only on the small, already-relevant subset, THEN the
    precise gpd.clip(). Each stage prints its own row count and timing
    so a long run is never silent."""
    if gdf.empty:
        return gdf

    n_raw = len(gdf)
    t0 = time.perf_counter()

    candidate_idx = list(gdf.sindex.query(clip_geom, predicate="intersects"))
    gdf = gdf.iloc[candidate_idx].copy()
    t1 = time.perf_counter()
    if label:
        print(f"[{label}] Bbox pre-filter: {n_raw:,} -> {len(gdf):,} rows ({t1-t0:.1f}s)")

    if gdf.empty:
        return gdf

    gdf["geometry"] = gdf.geometry.apply(make_valid)
    # FIX: make_valid() can legitimately repair a pathological/self-
    # intersecting geometry by returning a GeometryCollection — which
    # can be technically "valid" and non-empty while containing a stray
    # LineString or Point alongside real polygon area. Left in, this
    # produces a mixed-geometry-type dataset that gpd.overlay() refuses
    # to process ("df2 contains mixed geometry types"). Since only area
    # matters here, dropping any non-Polygon/MultiPolygon fragments is
    # safe — they contribute ~zero real area regardless of their origin.
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty
              & gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    t2 = time.perf_counter()
    if label:
        print(f"[{label}] Validated: {len(gdf):,} rows kept ({t2-t1:.1f}s)")

    if gdf.empty:
        return gdf

    clipped = gpd.clip(gdf, clip_geom)
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()].copy()
    if simplify_m > 0 and not clipped.empty:
        clipped["geometry"] = clipped.geometry.simplify(simplify_m, preserve_topology=True)
        clipped = clipped[~clipped.geometry.is_empty].copy()
    t3 = time.perf_counter()
    if label:
        print(f"[{label}] Clipped: {len(gdf):,} -> {len(clipped):,} features "
              f"{f'(simplified at {simplify_m}m) ' if simplify_m > 0 else ''}({t3-t2:.1f}s)  "
              f"[total {t3-t0:.1f}s]")

    return clipped.reset_index(drop=True)


def _overlay_batch_worker(batch_tiles: gpd.GeoDataFrame, batch_source: gpd.GeoDataFrame) -> pd.Series:
    """Top-level worker for parallel batch overlay — must be module-level
    so ProcessPoolExecutor can pickle it."""
    if batch_source.empty:
        return pd.Series(dtype=float)
    batch_overlay = gpd.overlay(batch_tiles, batch_source, how="intersection", keep_geom_type=False)
    if len(batch_overlay) == 0:
        return pd.Series(dtype=float)
    return batch_overlay.groupby("h3_cell")["geometry"].apply(lambda g: g.area.sum())


def _dissolve_cell_worker(core_bounds: tuple, buffer_m: float,
                           cell_canopy: gpd.GeoDataFrame, crs) -> gpd.GeoSeries:
    """Top-level (module-level, so ProcessPoolExecutor/spawn can pickle
    it) worker: dissolves self-overlap within one spatial grid cell.
    cell_canopy is already spatially pre-filtered to this cell's
    buffered extent on the main process before dispatch. Returns only
    the resulting pieces whose centroid falls in the cell's CORE
    (non-buffered) region, so adjacent cells' overlapping buffer zones
    never produce duplicate output for the same real polygon."""
    if cell_canopy.empty:
        return gpd.GeoSeries([], crs=crs)
    core_cell = box(*core_bounds)
    merged = cell_canopy.geometry.union_all()
    exploded = gpd.GeoSeries([merged], crs=crs).explode(index_parts=False).reset_index(drop=True)
    keep_mask = exploded.centroid.within(core_cell) & exploded.geom_type.isin(["Polygon", "MultiPolygon"])
    return exploded[keep_mask].reset_index(drop=True)


def _batched_dissolve(canopy: gpd.GeoDataFrame, grid_cell_m: float = 400,
                       buffer_m: float = 20, n_workers: int = None) -> gpd.GeoDataFrame:
    """FIX: replaces the old "cheap all-pairs overlap check, then maybe
    one monolithic union_all()" pattern. For a genuinely large canopy
    dataset (Bari: 540K polygons, 95K overlapping pairs), the "cheap"
    check itself took 892s, and the subsequent full dissolve would have
    taken longer still — an all-pairs style check is not actually cheap
    at this scale, and it was only ever deciding whether to run
    something even more expensive.

    Canopy self-overlap is inherently LOCAL: a fragment on one side of
    a city can never overlap one on the other side. This partitions the
    canopy extent into a grid (mirroring the same GRID_CELL_M pattern
    already used in canopy_extract_rgb.py's own dissolve), dissolves
    each cell independently and in parallel, and de-duplicates
    boundary-straddling overlaps via a buffer-and-centroid-filter — the
    same correctness pattern, just applied here instead of at
    extraction time. This always runs (no separate pre-check needed):
    a cell with no real overlap dissolves through essentially
    unchanged, and cheaply, since it's only operating on that cell's
    small local subset either way."""
    if canopy.empty:
        return canopy

    minx, miny, maxx, maxy = canopy.total_bounds
    cols = np.arange(minx, maxx, grid_cell_m)
    rows = np.arange(miny, maxy, grid_cell_m)
    sindex = canopy.sindex
    crs = canopy.crs

    cell_jobs = []
    for cx in cols:
        for cy in rows:
            core_bounds = (cx, cy, cx + grid_cell_m, cy + grid_cell_m)
            buffered_cell = box(*core_bounds).buffer(buffer_m)
            candidate_idx = list(sindex.query(buffered_cell, predicate="intersects"))
            if not candidate_idx:
                continue
            cell_jobs.append((core_bounds, canopy.iloc[candidate_idx][["geometry"]]))

    if not cell_jobs:
        return canopy.iloc[0:0]

    n_workers = n_workers or min(8, os.cpu_count() or 1)
    t_start = time.perf_counter()
    print(f"[canopy] Dissolving {len(canopy):,} polygons across {len(cell_jobs)} grid cells "
          f"({n_workers} workers)...")

    dissolved_parts = []
    if n_workers <= 1 or len(cell_jobs) <= 1:
        for i, (core_bounds, cell_gdf) in enumerate(cell_jobs):
            dissolved_parts.append(_dissolve_cell_worker(core_bounds, buffer_m, cell_gdf, crs))
            print(f"  [dissolve] cell {i+1}/{len(cell_jobs)}  ({time.perf_counter()-t_start:.0f}s elapsed)")
    else:
        spawn_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=spawn_ctx) as pool:
            futures = {
                pool.submit(_dissolve_cell_worker, core_bounds, buffer_m, cell_gdf, crs): i
                for i, (core_bounds, cell_gdf) in enumerate(cell_jobs)
            }
            completed = 0
            for f in as_completed(futures):
                dissolved_parts.append(f.result())
                completed += 1
                print(f"  [dissolve] cell {completed}/{len(cell_jobs)}  "
                      f"({time.perf_counter()-t_start:.0f}s elapsed, {n_workers} workers)")

    dissolved_parts = [p for p in dissolved_parts if not p.empty]
    if not dissolved_parts:
        return canopy.iloc[0:0]

    result_geom = pd.concat(dissolved_parts, ignore_index=True)
    result = gpd.GeoDataFrame(geometry=result_geom, crs=crs).reset_index(drop=True)
    print(f"[canopy] Dissolved: {len(canopy):,} -> {len(result):,} polygons "
          f"({time.perf_counter()-t_start:.0f}s total)")
    return result


TOTAL_STAGES = 9

def _stage(n: int, label: str, city: str):
    """Prints a discrete stage counter (n/TOTAL_STAGES) rather than a
    precise time-based percentage. Given how wildly different each
    stage's actual duration is city-to-city (a validate step can take
    0.4s or 400s depending on dataset size), a true time-weighted
    percentage would be more misleading than useful. This just answers
    "how many stages are left," which is the concrete thing actually
    being asked for."""
    pct = round(100 * (n - 1) / TOTAL_STAGES)
    print(f"\n[{city}] --- Stage {n}/{TOTAL_STAGES} (~{pct}% through pipeline): {label} ---")


def compute_canopy_compliance(tiles: gpd.GeoDataFrame, con, city: str) -> gpd.GeoDataFrame:
    """Compute canopy coverage and 3-30-300 compliance for each hex tile."""
    base_city = strip_source_suffix(city)

    # Computed FIRST, before loading anything else — only depends on
    # tiles, which is already available. Every load below immediately
    # pre-filters against this extent before doing anything expensive.
    tiles_extent = tiles.geometry.union_all().buffer(50)

    _stage(2, "Load + validate + clip canopy", city)
    t_load = time.perf_counter()
    canopy_raw = _load_and_prepare_canopy(con, city).to_crs(CRS_METRIC)
    print(f"[{city}] Canopy loaded: {len(canopy_raw):,} rows ({time.perf_counter()-t_load:.1f}s)")
    canopy = _prefilter_validate_clip(canopy_raw, tiles_extent, label="canopy")
    del canopy_raw

    _stage(3, "Load + validate + clip buildings", city)
    t_load = time.perf_counter()
    buildings_raw = load_table(con, "raw", f"{base_city}_buildings").to_crs(CRS_METRIC)
    print(f"[{city}] Buildings loaded: {len(buildings_raw):,} rows ({time.perf_counter()-t_load:.1f}s)")
    buildings = _prefilter_validate_clip(buildings_raw, tiles_extent, label="buildings")
    del buildings_raw

    _stage(4, "Load + validate + clip water", city)
    t_load = time.perf_counter()
    try:
        water_raw = load_table(con, "raw", f"{base_city}_water").to_crs(CRS_METRIC)
        print(f"[{city}] Water loaded: {len(water_raw):,} rows ({time.perf_counter()-t_load:.1f}s)")
    except Exception:
        print(f"[water] WARNING: raw.{city}_water table not found despite syncing! Using empty fallback.")
        water_raw = gpd.GeoDataFrame(columns=["geometry"], crs=CRS_METRIC)
    water = _prefilter_validate_clip(water_raw, tiles_extent, simplify_m=1.0, label="water")
    del water_raw

    _stage(5, "Dissolve canopy self-overlap", city)
    canopy = _batched_dissolve(canopy)

    print(f"[{city}] Computing canopy compliance for {len(tiles):,} hexes...")

    N_WORKERS = min(8, os.cpu_count() or 1)

    def _batched_overlay_area(source_gdf, label, batch_size=200):
        if source_gdf.empty:
            return pd.Series(dtype=float, name=f"{label}_m2")

        t_start = time.perf_counter()
        area_parts = []

        batch_jobs = []
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles.iloc[i:i + batch_size][["h3_cell", "geometry"]]
            batch_extent = box(*batch_tiles.total_bounds).buffer(10)
            candidate_idx = list(source_gdf.sindex.query(batch_extent, predicate="intersects"))
            if not candidate_idx:
                continue
            batch_source = source_gdf.iloc[candidate_idx][["geometry"]]
            batch_jobs.append((batch_tiles, batch_source))

        if not batch_jobs:
            return pd.Series(dtype=float, name=f"{label}_m2")

        print(f"  [{label}] {len(batch_jobs)} batches queued...")

        if N_WORKERS <= 1 or len(batch_jobs) <= 1:
            for i, (bt, bs) in enumerate(batch_jobs):
                area_parts.append(_overlay_batch_worker(bt, bs))
                print(f"  [{label}] batch {i+1}/{len(batch_jobs)}  "
                      f"({time.perf_counter()-t_start:.0f}s elapsed)")
        else:
            spawn_ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=spawn_ctx) as pool:
                futures = {pool.submit(_overlay_batch_worker, bt, bs): idx
                           for idx, (bt, bs) in enumerate(batch_jobs)}
                completed = 0
                # FIX: was every-10-batches. With widely-varying batch
                # sizes/complexity, that could leave a run silent for a
                # long stretch with no visible sign of progress at all.
                # Printing every completed batch is cheap and keeps this
                # actually monitorable in real time.
                for f in as_completed(futures):
                    area_parts.append(f.result())
                    completed += 1
                    elapsed = time.perf_counter() - t_start
                    print(f"  [{label}] batch {completed}/{len(batch_jobs)}  "
                          f"({elapsed:.0f}s elapsed, {N_WORKERS} workers)")

        area_parts = [s for s in area_parts if not s.empty]
        if not area_parts:
            return pd.Series(dtype=float, name=f"{label}_m2")
        return pd.concat(area_parts).groupby(level=0).sum().rename(f"{label}_m2")

    _stage(6, "Building footprint overlay", city)
    bld_area = _batched_overlay_area(buildings, "bld")
    print(f"[{city}] Building footprint areas computed for {len(bld_area):,} hexes.")

    _stage(7, "Water body overlay", city)
    water_area = _batched_overlay_area(water, "water")
    print(f"[{city}] Water body areas computed for {len(water_area):,} hexes.")

    _stage(8, "Canopy overlay", city)
    canopy_area = _batched_overlay_area(canopy, "canopy")
    print(f"[{city}] Canopy areas computed for {len(canopy_area):,} hexes.")
    print(f"[{city}] Merging building, water, and canopy areas into hex tiles...")

    tiles = tiles.copy()
    tiles["hex_area_m2"] = tiles.geometry.area.round(2)
    tiles["bld_m2"]      = tiles["h3_cell"].map(bld_area).fillna(0.0).round(2)
    tiles["water_m2"]    = tiles["h3_cell"].map(water_area).fillna(0.0).round(2)
    tiles["ground_area"] = (tiles["hex_area_m2"] - tiles["water_m2"]).clip(lower=0).round(2)
    tiles["canopy_m2"]  = tiles["h3_cell"].map(canopy_area).fillna(0.0).round(2)
    tiles["canopy_pct"] = (
        tiles["canopy_m2"] / tiles["ground_area"] * 100
    ).where(tiles["ground_area"] > 0, 0.0).round(2)
    tiles["compliant"]  = tiles["canopy_pct"] >= CANOPY_THRESHOLD
    return tiles


def compute_civic_compliance(civici_wgs: gpd.GeoDataFrame, hex_results: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Join each civic address point to its hex's compliance flag."""
    hex_compliance = hex_results.set_index("h3_cell")[["compliant", "canopy_pct"]]
    civici_out = civici_wgs.merge(
        hex_compliance, left_on="h3_cell", right_index=True, how="left"
    )
    civici_out["compliant"] = civici_out["compliant"].fillna(False)
    return civici_out


def save_results(hex_gdf: gpd.GeoDataFrame, civici_gdf: gpd.GeoDataFrame, con, city: str) -> None:
    """Persist hex-level and civico-level results, log stats, upsert shared CSV."""
    out_dir = DATA_DIR / city
    out_dir.mkdir(parents=True, exist_ok=True)

    hex_gpkg_path = out_dir / f"canopy_compliance_{city}.gpkg"
    hex_gdf.to_file(hex_gpkg_path, driver="GPKG", layer="canopy_compliance")
    gdf_to_duckdb(con, hex_gdf, f"processed.canopy_compliance_{city}")

    civici_gpkg_path = out_dir / f"canopy_compliance_{city}_by_civico.gpkg"
    civici_gdf.to_file(civici_gpkg_path, driver="GPKG", layer="civici_compliance")
    gdf_to_duckdb(con, civici_gdf, f"processed.canopy_compliance_{city}_by_civico")

    n_hex, nc_hex = len(hex_gdf), int(hex_gdf["compliant"].sum())
    n_civ = len(civici_gdf)
    nc_civ = int(civici_gdf["compliant"].sum())
    mean_canopy = hex_gdf["canopy_pct"].mean()

    log.info("[%s] hexes: %d | compliant: %d (%.1f%%) | mean canopy: %.1f%%",
             city, n_hex, nc_hex, nc_hex/n_hex*100 if n_hex else 0, mean_canopy)
    log.info("[%s] civici: %d | in compliant hex: %d (%.1f%%)",
             city, n_civ, nc_civ, nc_civ/n_civ*100 if n_civ else 0)

    summary_row = {
        "city": city,
        "n_hexes_total": n_hex,
        "n_hexes_compliant": nc_hex,
        "pct_hexes_compliant": round(100 * nc_hex / max(n_hex, 1), 2),
        "mean_canopy_pct": round(float(mean_canopy), 2) if n_hex else None,
        "n_civici_total": n_civ,
        "n_civici_in_compliant_hex": nc_civ,
        "pct_civici_in_compliant_hex": round(100 * nc_civ / max(n_civ, 1), 2),
        "res_hex": RES_HEX,
        "canopy_threshold_pct": CANOPY_THRESHOLD,
    }

    if SUMMARY_CSV.exists():
        summary_df = pd.read_csv(SUMMARY_CSV)
        summary_df = summary_df[summary_df["city"] != city]
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([summary_row])

    summary_df.to_csv(SUMMARY_CSV, index=False)
    log.info("[%s] Summary stats upserted -> %s (%d cities total)",
             city, SUMMARY_CSV, len(summary_df))


def run(city: str) -> gpd.GeoDataFrame:
    """Full pipeline: build hex grid -> compute canopy compliance ->
    compute civic-level compliance -> save all results."""
    con = connect_db_with_retry()
    t0  = time.perf_counter()

    _stage(1, "Build H3 hex grid", city)
    tiles, civici_wgs = build_hex_layer(con, city)
    hex_result = compute_canopy_compliance(tiles, con, city)
    civici_result = compute_civic_compliance(civici_wgs, hex_result)
    _stage(9, "Save results", city)
    save_results(hex_result, civici_result, con, city)
    print(f"[{city}] Done in {time.perf_counter() - t0:.1f}s")
    print(f"30% canopy compliance for {city} is {hex_result['compliant'].mean() * 100:.1f}% of hexes")
    print(f"30% canopy compliance for {city} is {civici_result['compliant'].mean() * 100:.1f}% of civic addresses")
    print(f"Results saved to {DATA_DIR / city} and processed schema in DuckDB")

    log.info("[%s] Total: %.1fs", city, time.perf_counter() - t0)
    con.close()
    return hex_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    run(args.city)