from __future__ import annotations

#=====================================================================
CITY = "bologna"
#=====================================================================

RES_HEX = 8          # H3 resolution 8 ≈ 0.6 km² hexagons — according to 3-30-300 handbook guidance
CANOPY_THRESHOLD = 30.0  # 30% canopy cover 

import os, sys, time, logging
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
import duckdb
from pathlib import Path
from shapely.geometry import Polygon
from shapely.validation import make_valid

project_root = Path("/Users/terezasaskova/Library/Mobile Documents/com~apple~CloudDocs/3-30-300-index")
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from config.config import (connect_db, load_table, gdf_to_duckdb,
                           get_city_cfg, get_output_paths,
                           CRS_WGS, CRS_METRIC)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def finite_geom(geom) -> bool:
    #Reject geometries with non-finite coordinates (NaN/Inf) that break spatial ops
    return np.isfinite(np.array(geom.bounds)).all()


def h3_to_gdf(cells: list[str]) -> gpd.GeoDataFrame:
    #Convert a list of H3 cell IDs to a GeoDataFrame of polygons in CRS_METRIC.
    #Drops any cells whose bounds contain non-finite values

    polys = [Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in cells]
    gdf = gpd.GeoDataFrame({"h3_cell": cells}, geometry=polys, crs=CRS_WGS).to_crs(CRS_METRIC)
    return gdf[gdf.geometry.apply(finite_geom)].reset_index(drop=True)


def build_hex_layer(con) -> gpd.GeoDataFrame:
    """Build the H3 hex grid for the city.

    - Assigns each civic address point (buildings_sn) to its H3 cell.
    - Creates hex polygons only for cells that contain at least one civic address,
      so uninhabited areas outside the urban fabric are excluded.
    - Clips hexes to the city boundary and drops empty/invalid results.
    Returns a GeoDataFrame with columns: h3_cell, civic_count, geometry (CRS_METRIC).
    """
    civici   = load_table(con, "raw", f"{CITY}_buildings_sn")
    boundary = load_table(con, "raw", f"{CITY}_boundary")
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

    log.info("Hex tiles after boundary clip: %d", len(tiles))
    return tiles


def compute_canopy_compliance(tiles: gpd.GeoDataFrame, con) -> gpd.GeoDataFrame:
    """Compute canopy coverage and 3-30-300 compliance for each hex tile.

    Canopy % is computed over pervious area (hex area minus building footprints),
    not raw hex area. This avoids penalising dense urban hexes where buildings
    physically prevent canopy growth — only the ground where trees *can* grow counts.

    Columns added to tiles:
      hex_area_m2      — total hex polygon area
      bld_m2           — building footprint area within hex
      ground_area.     — hex_area_m2 - bld_m2 (clamped to 0)
      canopy_m2        — canopy area within hex
      canopy_pct       — canopy_m2 / ground_area * 100
      compliant        — True if canopy_pct >= CANOPY_THRESHOLD (30%)
    """
    canopy = load_table(con, "preprocessing", f"canopy_{CITY}", crs=CRS_WGS).to_crs(CRS_METRIC)
    canopy["geometry"] = canopy.geometry.apply(make_valid)
    canopy = canopy[canopy.geometry.is_valid & ~canopy.geometry.is_empty].reset_index(drop=True)

    buildings = load_table(con, "raw", f"{CITY}_buildings").to_crs(CRS_METRIC)
    buildings["geometry"] = buildings.geometry.apply(make_valid)
    buildings = buildings[buildings.geometry.is_valid & ~buildings.geometry.is_empty]

    bld_overlay = gpd.overlay(tiles[["h3_cell", "geometry"]], buildings[["geometry"]],
                               how="intersection", keep_geom_type=False)
    bld_area = (
        bld_overlay.groupby("h3_cell")["geometry"]
        .apply(lambda g: g.union_all().area)
        .rename("bld_m2")
        if len(bld_overlay) > 0
        else pd.Series(dtype=float, name="bld_m2")
    )

    overlay = gpd.overlay(tiles[["h3_cell", "geometry"]], canopy[["geometry"]],
                           how="intersection", keep_geom_type=False)
    canopy_area = (
        overlay.groupby("h3_cell")["geometry"]
        .apply(lambda g: g.union_all().area)
        .rename("canopy_m2")
        if len(overlay) > 0
        else pd.Series(dtype=float, name="canopy_m2")
    )

    tiles = tiles.copy()
    tiles["hex_area_m2"]      = tiles.geometry.area.round(2)
    tiles["bld_m2"]           = tiles["h3_cell"].map(bld_area).fillna(0.0).round(2)
    tiles["ground_area"] = (tiles["hex_area_m2"] - tiles["bld_m2"]).clip(lower=0).round(2)
    tiles["canopy_m2"]        = tiles["h3_cell"].map(canopy_area).fillna(0.0).round(2)
    tiles["canopy_pct"]       = (
        tiles["canopy_m2"] / tiles["ground_area"] * 100
    ).where(tiles["ground_area"] > 0, 0.0).round(2)
    tiles["compliant"]        = tiles["canopy_pct"] >= CANOPY_THRESHOLD
    return tiles


def save_results(gdf: gpd.GeoDataFrame, con) -> None:
    """Persist results to GeoPackage and DuckDB, then log summary statistics."""
    gpkg_path = f"data/{CITY}/canopy_compliance_{CITY}.gpkg"
    Path(gpkg_path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(gpkg_path, driver="GPKG", layer="canopy_compliance")
    gdf_to_duckdb(con, gdf, f"processed.canopy_compliance_{CITY}")

    n, nc = len(gdf), int(gdf["compliant"].sum())
    log.info("hexes: %d | compliant: %d (%.1f%%) | mean canopy: %.1f%%",
             n, nc, nc/n*100 if n else 0, gdf["canopy_pct"].mean())


def run(city: str = CITY) -> gpd.GeoDataFrame:
    """Full pipeline: build hex grid → compute canopy compliance → save results."""
    con = connect_db()
    t0  = time.perf_counter()

    tiles  = build_hex_layer(con)
    result = compute_canopy_compliance(tiles, con)
    save_results(result, con)

    log.info("Total: %.1fs", time.perf_counter() - t0)
    con.close()
    return result


if __name__ == "__main__":
    run()