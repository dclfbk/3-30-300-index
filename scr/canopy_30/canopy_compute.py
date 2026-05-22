from __future__ import annotations

CITY = "bologna"
RES_HEX = 8
CANOPY_THRESHOLD = 30.0

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

from config.config import connect_db, load_table, gdf_to_duckdb, get_city_cfg, get_output_paths, CRS_WGS, CRS_METRIC

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def finite_geom(geom) -> bool:
    return np.isfinite(np.array(geom.bounds)).all()

def h3_to_gdf(cells: list[str]) -> gpd.GeoDataFrame:
    polys = [Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in cells]
    gdf = gpd.GeoDataFrame({"h3_cell": cells}, geometry=polys, crs=CRS_WGS).to_crs(CRS_METRIC)
    return gdf[gdf.geometry.apply(finite_geom)].reset_index(drop=True)


# ── build hex grid with civic counts ─────────────────────────────────────────

def build_hex_layer(con) -> gpd.GeoDataFrame:
    civici   = load_table(con, "raw", f"{CITY}_buildings_sn")
    boundary = load_table(con, "raw", f"{CITY}_boundary")
    boundary_geom = boundary.geometry.iloc[0]

    # assign H3 cells (needs WGS84)
    civici_wgs = civici.to_crs(CRS_WGS)
    civici_wgs["h3_cell"] = civici_wgs.geometry.apply(
        lambda g: h3.latlng_to_cell(g.y, g.x, RES_HEX)
    )

    civic_counts = civici_wgs.groupby("h3_cell").size().rename("civic_count")
    tiles = h3_to_gdf(civic_counts.index.tolist())
    tiles["civic_count"] = tiles["h3_cell"].map(civic_counts).fillna(0).astype(int)

    # clip to boundary
    tiles = tiles[tiles.geometry.intersects(boundary_geom)].copy()
    tiles["geometry"] = tiles.geometry.intersection(boundary_geom)
    tiles = tiles[~tiles.geometry.is_empty & tiles.geometry.apply(finite_geom)].reset_index(drop=True)

    log.info("Hex tiles after boundary clip: %d", len(tiles))
    return tiles


# ── canopy compliance per hex ─────────────────────────────────────────────────

def compute_canopy_compliance(tiles: gpd.GeoDataFrame, con) -> gpd.GeoDataFrame:
    canopy = load_table(con, "preprocessing", f"canopy_{CITY}", crs=CRS_WGS).to_crs(CRS_METRIC)
    canopy["geometry"] = canopy.geometry.apply(make_valid)
    canopy = canopy[canopy.geometry.is_valid & ~canopy.geometry.is_empty].reset_index(drop=True)

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
    tiles["hex_area_m2"] = tiles.geometry.area.round(2)
    tiles["canopy_m2"]   = tiles["h3_cell"].map(canopy_area).fillna(0.0).round(2)
    tiles["canopy_pct"]  = (tiles["canopy_m2"] / tiles["hex_area_m2"] * 100).where(
        tiles["hex_area_m2"] > 0, 0.0).round(2)
    tiles["compliant"]   = tiles["canopy_pct"] >= CANOPY_THRESHOLD
    return tiles


# ── save ──────────────────────────────────────────────────────────────────────

def save_results(gdf: gpd.GeoDataFrame, con) -> None:
    gpkg_path = f"data/{CITY}/canopy_compliance_{CITY}.gpkg"
    Path(gpkg_path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(gpkg_path, driver="GPKG", layer="canopy_compliance")
    gdf_to_duckdb(con, gdf, f"processed.canopy_compliance_{CITY}")

    n, nc = len(gdf), int(gdf["compliant"].sum())
    log.info("hexes: %d | compliant: %d (%.1f%%) | mean canopy: %.1f%%",
             n, nc, nc/n*100 if n else 0, gdf["canopy_pct"].mean())


# ── run ───────────────────────────────────────────────────────────────────────

def run(city: str = CITY) -> gpd.GeoDataFrame:
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